#!/usr/bin/env python3
"""Sweet Maria's Green Coffee Scraper — uses Shopify JSON API + plain HTTP.

Extracts: name, price, stock, total score, cupping dimension scores,
flavor profile scores, overview, cupping notes, farm notes, and specs.

Run with --help for usage.
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime
from html import unescape as html_unescape
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "coffees.db"
BASE_URL = "https://www.sweetmarias.com"
COLLECTION_JSON = BASE_URL + "/collections/green-coffee/products.json"
DELAY = 3  # seconds between requests
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def _http_get(url):
    """Fetch a URL and return the response body as a string."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode()


def find_coffee(conn, query: str, cmd: str):
    """Look up a coffee by URL or name. Returns row or None (prints disambiguation/not found)."""
    if query.startswith("http"):
        row = conn.execute(
            "SELECT * FROM coffees WHERE url = ? OR url LIKE ?", (query, f"%{query}%")
        ).fetchone()
        if not row:
            print(f"No coffee found matching '{query}'")
        return row
    rows = conn.execute(
        "SELECT * FROM coffees WHERE name LIKE ? ORDER BY total_score DESC",
        (f"%{query}%",),
    ).fetchall()
    if not rows:
        print(f"No coffee found matching '{query}'")
        return None
    if len(rows) == 1:
        return rows[0]
    print(f"Multiple coffees match '{query}':\n")
    for r in rows:
        print(f"  {r['name']}")
        print(f'    ./scrape_sm.py {cmd} "{r["url"]}"')
        print()
    return None


# --- Shared constants and helpers for flavor analysis ---

CUPPING_COLS = [
    "dry_fragrance",
    "wet_aroma",
    "brightness",
    "flavor",
    "body",
    "finish",
    "sweetness",
    "clean_cup",
    "complexity",
    "uniformity",
]
FLAVOR_COLS = [
    "fl_floral",
    "fl_honey",
    "fl_sugars",
    "fl_caramel",
    "fl_fruits",
    "fl_citrus",
    "fl_berry",
    "fl_cocoa",
    "fl_nuts",
    "fl_rustic",
    "fl_spice",
    "fl_body",
]
ALL_COLS = CUPPING_COLS + FLAVOR_COLS
DIMS = [
    ("Dry Fragrance", "dry_fragrance"),
    ("Wet Aroma", "wet_aroma"),
    ("Brightness", "brightness"),
    ("Flavor", "flavor"),
    ("Body", "body"),
    ("Finish", "finish"),
    ("Sweetness", "sweetness"),
    ("Clean Cup", "clean_cup"),
    ("Complexity", "complexity"),
    ("Uniformity", "uniformity"),
    ("Floral", "fl_floral"),
    ("Honey", "fl_honey"),
    ("Sugars", "fl_sugars"),
    ("Caramel", "fl_caramel"),
    ("Fruits", "fl_fruits"),
    ("Citrus", "fl_citrus"),
    ("Berry", "fl_berry"),
    ("Cocoa", "fl_cocoa"),
    ("Nuts", "fl_nuts"),
    ("Rustic", "fl_rustic"),
    ("Spice", "fl_spice"),
    ("Body(fl)", "fl_body"),
]
DIM_NAMES = [d[0] for d in DIMS]
# Max values per dimension for normalization: cupping scores are 0-10, flavor profile scores are 0-5
SCALES = [10] * 10 + [5] * 12


def _is_decaf(row):
    return "decaf" in (row["name"] or "").lower()


def _matches_exclude(row, excludes):
    """Check if a row matches any exclude pattern (URL or case-insensitive name keyword)."""
    name = (row["name"] or "").lower()
    for ex in excludes:
        if ex.startswith("http"):
            if row["url"] == ex:
                return True
        else:
            if ex.lower() in name:
                return True
    return False


def _is_blend(row):
    """Check if a coffee is a blend (centroid-like, not useful for distance)."""
    name = (row["name"] or "").lower()
    cultivar = (row["cultivar"] or "").lower()
    return "blend" in name or "varies" == cultivar or "workshop" in name


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS coffees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE NOT NULL,
        name TEXT,
        price REAL,
        in_stock INTEGER,
        total_score REAL,
        -- Cupping scores (radar chart)
        dry_fragrance REAL, wet_aroma REAL, brightness REAL,
        flavor REAL, body REAL, finish REAL, sweetness REAL,
        clean_cup REAL, complexity REAL, uniformity REAL,
        -- Flavor profile (polar chart)
        fl_floral REAL, fl_honey REAL, fl_sugars REAL, fl_caramel REAL,
        fl_fruits REAL, fl_citrus REAL, fl_berry REAL, fl_cocoa REAL,
        fl_nuts REAL, fl_rustic REAL, fl_spice REAL, fl_body REAL,
        -- Text
        overview TEXT, cupping_notes TEXT, farm_notes TEXT,
        -- Specs
        origin TEXT, processing TEXT, cultivar TEXT, grade TEXT,
        appearance TEXT, roast_recommendations TEXT, type TEXT,
        espresso_recommended INTEGER, farm_gate INTEGER,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS tried (
        url TEXT PRIMARY KEY,
        rating TEXT NOT NULL DEFAULT '0',  -- '+', '0', '-'
        notes TEXT,
        tried_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS stock_events (
        url TEXT NOT NULL,
        event TEXT NOT NULL,  -- 'appeared', 'in_stock', 'out_of_stock', 'removed'
        observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_events_url ON stock_events(url)")
    conn.commit()
    _migrate_urls(conn)
    return conn


def _migrate_urls(conn):
    """Migrate old Magento URLs (.html) to new Shopify format (/products/)."""
    old_urls = conn.execute(
        "SELECT url FROM coffees WHERE url LIKE '%.html'"
    ).fetchall()
    if not old_urls:
        return
    count = 0
    for (old_url,) in old_urls:
        # https://www.sweetmarias.com/slug.html -> https://www.sweetmarias.com/products/slug
        slug = old_url.replace(BASE_URL + "/", "").replace(".html", "")
        new_url = f"{BASE_URL}/products/{slug}"
        try:
            conn.execute("UPDATE coffees SET url = ? WHERE url = ?", (new_url, old_url))
            count += 1
        except sqlite3.IntegrityError:
            # New URL already exists (duplicate) — skip
            pass
    # Also migrate tried table
    old_tried = conn.execute("SELECT url FROM tried WHERE url LIKE '%.html'").fetchall()
    for (old_url,) in old_tried:
        slug = old_url.replace(BASE_URL + "/", "").replace(".html", "")
        new_url = f"{BASE_URL}/products/{slug}"
        try:
            conn.execute("UPDATE tried SET url = ? WHERE url = ?", (new_url, old_url))
        except sqlite3.IntegrityError:
            pass
    # And stock_events
    conn.execute(
        """UPDATE stock_events SET url = REPLACE(
            REPLACE(url, '.html', ''),
            'https://www.sweetmarias.com/',
            'https://www.sweetmarias.com/products/'
        ) WHERE url LIKE '%.html'"""
    )
    # And roasts (if table exists)
    try:
        conn.execute(
            """UPDATE roasts SET bean_url = REPLACE(
                REPLACE(bean_url, '.html', ''),
                'https://www.sweetmarias.com/',
                'https://www.sweetmarias.com/products/'
            ) WHERE bean_url LIKE '%.html'"""
        )
    except sqlite3.OperationalError:
        pass  # roasts table may not exist
    conn.commit()
    if count:
        logger.info("url migration migrated %d URLs to new format", count)


def get_catalog():
    """Fetch all products from the Shopify JSON API. Returns list of dicts with url, handle, available, title, price."""
    products = []
    page = 1
    while True:
        url = f"{COLLECTION_JSON}?limit=250&page={page}"
        logger.info("catalog fetch page %d: %s", page, url)
        raw = _http_get(url)
        data = json.loads(raw)
        batch = data.get("products", [])
        if not batch:
            break
        for p in batch:
            available = any(v["available"] for v in p["variants"])
            price_str = p["variants"][0]["price"] if p["variants"] else "0"
            products.append(
                {
                    "url": f"{BASE_URL}/products/{p['handle']}",
                    "handle": p["handle"],
                    "available": available,
                    "title": p["title"],
                    "price": float(price_str),
                }
            )
        page += 1
    logger.info("catalog fetch complete: %d products total", len(products))
    return products


def _parse_chart_values(raw):
    """Parse 'Key:val,Key:val,...' into a dict of lowercase_key -> float."""
    result = {}
    if not raw:
        return result
    for pair in raw.split(","):
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        try:
            result[k.strip().lower().replace(" ", "_")] = float(v)
        except ValueError:
            cleaned = v.strip().replace("..", ".")
            try:
                result[k.strip().lower().replace(" ", "_")] = float(cleaned)
            except ValueError:
                result[k.strip().lower().replace(" ", "_")] = None
    return result


def scrape_product(url):
    """Fetch a product page via HTTP and extract all data using regex."""
    logger.debug("scraping product %s", url)
    html = _http_get(url)

    # Name from <h1>
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL)
    name = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else None
    if name:
        name = html_unescape(name)
    if not name:
        logger.warning("scrape failed no name found for %s", url)
        return None

    # Score from data-chart-score
    m = re.search(r'data-chart-score="([^"]+)"', html)
    total_score = float(m.group(1)) if m else None

    # Cupping chart values
    m = re.search(r'data-chart-id="cupping-chart"[^>]*data-chart-value="([^"]+)"', html)
    cupping = _parse_chart_values(m.group(1) if m else None)

    # Flavor chart values
    m = re.search(r'data-chart-id="flavor-chart"[^>]*data-chart-value="([^"]+)"', html)
    flavors = _parse_chart_values(m.group(1) if m else None)

    # Stock — check for availability in schema.org or "In stock" text
    in_stock = "InStock" in html or ">In stock<" in html

    # Price from schema.org JSON-LD (most reliable)
    m = re.search(r'"price"\s*:\s*([0-9]+\.?[0-9]*)', html)
    price = float(m.group(1)) if m else None
    # The schema price for the 1 LB variant is what we want
    # Variant JSON has price in cents — check schema first
    m = re.search(
        r'"@type"\s*:\s*"Offer"[^}]*"name"\s*:\s*"1 LB"[^}]*"price"\s*:\s*([0-9.]+)',
        html,
    )
    if m:
        price = float(m.group(1))

    # Specs from <th>/<td> pairs in the Technical Specifications section
    specs = {}
    specs_idx = html.find("Technical Specifications")
    if specs_idx > 0:
        chunk = html[specs_idx : specs_idx + 5000]
        rows = re.findall(
            r"<th[^>]*>\s*(.*?)\s*</th>\s*<td[^>]*>\s*(.*?)\s*</td>",
            chunk,
            re.DOTALL,
        )
        for th, td in rows:
            key = re.sub(r"<[^>]+>", "", th).strip()
            val = re.sub(r"<[^>]+>", "", td).strip()
            if key:
                specs[key.lower()] = val

    # Cupping notes — find paragraphs after "Full Cupping Notes"
    cupping_notes = None
    cn_idx = html.find("Full Cupping Notes")
    if cn_idx > 0:
        after = html[cn_idx : cn_idx + 5000]
        paragraphs = re.findall(r"<p>(.*?)</p>", after, re.DOTALL)
        if paragraphs:
            cupping_notes = re.sub(r"<[^>]+>", "", paragraphs[0]).strip()

    # Farm notes — find paragraphs after "Origin" section
    farm_notes = None
    fn_idx = html.find("Origin &amp; Farm Notes")
    if fn_idx < 0:
        fn_idx = html.find("Origin & Farm Notes")
    if fn_idx > 0:
        after = html[fn_idx : fn_idx + 5000]
        paragraphs = re.findall(r"<p>(.*?)</p>", after, re.DOTALL)
        if paragraphs:
            farm_notes = re.sub(r"<[^>]+>", "", paragraphs[0]).strip()

    # Overview — the short description from body_html area
    overview = None
    m = re.search(r'data-content-type="text"[^>]*><p>(.*?)</p>', html, re.DOTALL)
    if m:
        overview = re.sub(r"<[^>]+>", "", m.group(1)).strip()

    return {
        "url": url,
        "name": name,
        "price": price,
        "in_stock": in_stock,
        "total_score": total_score,
        # Cupping
        "dry_fragrance": cupping.get("dry_fragrance"),
        "wet_aroma": cupping.get("wet_aroma"),
        "brightness": cupping.get("brightness"),
        "flavor": cupping.get("flavor"),
        "body": cupping.get("body"),
        "finish": cupping.get("finish"),
        "sweetness": cupping.get("sweetness"),
        "clean_cup": cupping.get("clean_cup"),
        "complexity": cupping.get("complexity"),
        "uniformity": cupping.get("uniformity"),
        # Flavor
        "fl_floral": flavors.get("floral"),
        "fl_honey": flavors.get("honey"),
        "fl_sugars": flavors.get("sugars"),
        "fl_caramel": flavors.get("caramel"),
        "fl_fruits": flavors.get("fruits"),
        "fl_citrus": flavors.get("citrus"),
        "fl_berry": flavors.get("berry"),
        "fl_cocoa": flavors.get("cocoa"),
        "fl_nuts": flavors.get("nuts"),
        "fl_rustic": flavors.get("rustic"),
        "fl_spice": flavors.get("spice"),
        "fl_body": flavors.get("body"),
        # Text
        "overview": overview,
        "cupping_notes": cupping_notes,
        "farm_notes": farm_notes,
        # Specs
        "processing": specs.get("processing"),
        "cultivar": specs.get("cultivar detail"),
        "grade": specs.get("grade"),
        "appearance": specs.get("appearance"),
        "roast_recommendations": specs.get("roast recommendations"),
        "type": specs.get("type"),
        "espresso_recommended": (
            "yes" in specs.get("recommended espresso", "").lower()
            if specs.get("recommended espresso")
            else None
        ),
        "farm_gate": (
            "yes" in specs.get("farm gate", "").lower()
            if specs.get("farm gate")
            else None
        ),
        "origin": specs.get("region"),
    }


def save_coffee(conn, data):
    cols = [k for k in data.keys() if k != "origin"]  # origin handled separately
    placeholders = ",".join(["?"] * (len(cols) + 1))  # +1 for origin
    col_names = ",".join(cols + ["origin"])
    values = [data[k] for k in cols] + [data.get("origin")]
    conn.execute(
        f"INSERT OR REPLACE INTO coffees ({col_names}, scraped_at) VALUES ({placeholders}, CURRENT_TIMESTAMP)",
        values,
    )
    conn.commit()


def _save_with_transition(conn, data):
    """Save coffee and log stock transition if status changed."""
    prev = conn.execute(
        "SELECT in_stock FROM coffees WHERE url = ?", (data["url"],)
    ).fetchone()
    save_coffee(conn, data)
    if prev is not None:
        was_in_stock = bool(prev[0])
        now_in_stock = bool(data["in_stock"])
        if was_in_stock and not now_in_stock:
            conn.execute(
                "INSERT INTO stock_events (url, event) VALUES (?, 'out_of_stock')",
                (data["url"],),
            )
            conn.commit()
        elif not was_in_stock and now_in_stock:
            conn.execute(
                "INSERT INTO stock_events (url, event) VALUES (?, 'in_stock')",
                (data["url"],),
            )
            conn.commit()


def main_single(url):
    conn = init_db()
    print(f"Scraping: {url}")
    try:
        data = scrape_product(url)
    except Exception as e:
        logger.error("scrape single failed for %s: %s", url, e)
        print("❌ Failed to scrape")
        conn.close()
        return
    if data:
        save_coffee(conn, data)
        print(f"\n✅ {data['name']}")
        print(
            f"   Price: ${data['price']} | Score: {data['total_score']} | Stock: {'✅' if data['in_stock'] else '❌'}"
        )
        print("\n   Cupping Scores:")
        for k in [
            "dry_fragrance",
            "wet_aroma",
            "brightness",
            "flavor",
            "body",
            "finish",
            "sweetness",
            "clean_cup",
            "complexity",
            "uniformity",
        ]:
            v = data.get(k)
            if v is not None:
                print(f"     {k.replace('_', ' ').title():15s}: {v}")
        print("\n   Flavor Profile:")
        for k in [
            "fl_floral",
            "fl_honey",
            "fl_sugars",
            "fl_caramel",
            "fl_fruits",
            "fl_citrus",
            "fl_berry",
            "fl_cocoa",
            "fl_nuts",
            "fl_rustic",
            "fl_spice",
            "fl_body",
        ]:
            v = data.get(k)
            if v is not None:
                print(f"     {k[3:].title():10s}: {v}")
    else:
        print("❌ Failed to scrape")
    conn.close()


def main_all(include_out_of_stock=False):
    conn = init_db()
    existing = set(r[0] for r in conn.execute("SELECT url FROM coffees").fetchall())
    prev_in_stock = set(
        r[0]
        for r in conn.execute("SELECT url FROM coffees WHERE in_stock = 1").fetchall()
    )

    # Fetch full catalog from Shopify JSON API
    catalog = get_catalog()
    if not catalog:
        print("❌ Catalog returned 0 products — API may be down. Aborting.")
        conn.close()
        return

    in_stock_urls = set(p["url"] for p in catalog if p["available"])
    all_catalog_urls = set(p["url"] for p in catalog)

    # Guard: if suspiciously few results compared to previous state
    if prev_in_stock and len(in_stock_urls) < len(prev_in_stock) * 0.3:
        print(
            f"⚠️  Catalog shows only {len(in_stock_urls)} in stock "
            f"(previously {len(prev_in_stock)}). Aborting stock transitions."
        )
        conn.close()
        return

    # Stock transitions: compare in-stock catalog against DB state
    newly_out = prev_in_stock - in_stock_urls
    newly_in = (in_stock_urls & existing) - prev_in_stock
    if newly_out:
        conn.executemany(
            "INSERT INTO stock_events (url, event) VALUES (?, 'out_of_stock')",
            [(u,) for u in newly_out],
        )
        conn.executemany(
            "UPDATE coffees SET in_stock = 0 WHERE url = ?",
            [(u,) for u in newly_out],
        )
        conn.commit()
        print(f"Out of stock: {len(newly_out)} coffees")
    if newly_in:
        conn.executemany(
            "INSERT INTO stock_events (url, event) VALUES (?, 'in_stock')",
            [(u,) for u in newly_in],
        )
        conn.executemany(
            "UPDATE coffees SET in_stock = 1 WHERE url = ?",
            [(u,) for u in newly_in],
        )
        conn.commit()
        print(f"Restocked: {len(newly_in)} coffees")

    # Determine which coffees to scrape (new ones only)
    if include_out_of_stock:
        new_urls = [u for u in all_catalog_urls if u not in existing]
    else:
        new_urls = [u for u in in_stock_urls if u not in existing]

    # Log 'appeared' for brand new coffees
    if new_urls:
        conn.executemany(
            "INSERT INTO stock_events (url, event) VALUES (?, 'appeared')",
            [(u,) for u in new_urls],
        )
        conn.commit()

    print(f"Found {len(in_stock_urls)} in stock, {len(new_urls)} new to scrape\n")

    for i, url in enumerate(new_urls, 1):
        print(f"[{i}/{len(new_urls)}] {url.split('/')[-1]}")
        try:
            data = scrape_product(url)
        except Exception as e:
            logger.warning("scrape failed for %s: %s", url, e)
            print("    ⚠️  Failed")
            time.sleep(DELAY)
            continue
        if data:
            _save_with_transition(conn, data)
            print(f"    ✅ {data['name']} — Score: {data['total_score']}")
        else:
            print("    ⚠️  Failed")
        time.sleep(DELAY)

    conn.close()
    print(f"\nDone! {DB_PATH}")


def show(no_decaf=False):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM coffees ORDER BY total_score DESC").fetchall()
    if no_decaf:
        rows = [r for r in rows if not _is_decaf(r)]
    print(f"\n{'=' * 70}")
    print(f" Sweet Maria's Coffee Database — {len(rows)} coffees")
    print(f"{'=' * 70}\n")
    for r in rows:
        print(f"  {r['name']} — Score: {r['total_score']} — ${r['price']}")
        scores = [
            f"Frag:{r['dry_fragrance']}",
            f"Aroma:{r['wet_aroma']}",
            f"Bright:{r['brightness']}",
            f"Flavor:{r['flavor']}",
            f"Body:{r['body']}",
            f"Finish:{r['finish']}",
            f"Sweet:{r['sweetness']}",
            f"Clean:{r['clean_cup']}",
            f"Complex:{r['complexity']}",
            f"Uniform:{r['uniformity']}",
        ]
        print(f"    {' | '.join(s for s in scores if 'None' not in s)}")
        print()
    conn.close()


def stock():
    """Show weekly stock velocity: arrivals, sellouts, restocks."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    events = conn.execute(
        "SELECT DISTINCT url, event, MIN(observed_at) as observed_at "
        "FROM stock_events GROUP BY url, event ORDER BY observed_at"
    ).fetchall()
    if not events:
        print("No stock events recorded yet. Run 'all' to start tracking.")
        conn.close()
        return

    # Bucket events by ISO week (counting distinct URLs per event type)
    weeks = {}
    for e in events:
        dt = datetime.fromisoformat(e["observed_at"].replace(" ", "T"))
        wk = dt.strftime("%Y-W%W")
        weeks.setdefault(wk, {"appeared": 0, "out_of_stock": 0, "in_stock": 0})
        if e["event"] in weeks[wk]:
            weeks[wk][e["event"]] += 1

    # Current state
    cur = conn.execute(
        "SELECT SUM(in_stock) as avail, COUNT(*) as total FROM coffees"
    ).fetchone()

    print("\n  ── Stock Velocity ──")
    print(f"  Currently: {cur['avail']} in stock / {cur['total']} total tracked\n")
    print(f"  {'Week':<9} {'New':>5} {'Sold Out':>9} {'Restocked':>10}")
    print(f"  {'─' * 9} {'─' * 5} {'─' * 9} {'─' * 10}")
    for wk in sorted(weeks):
        w = weeks[wk]
        print(
            f"  {wk:<9} {w['appeared']:>5} {w['out_of_stock']:>9} {w['in_stock']:>10}"
        )
    print()
    conn.close()


def detail(query: str):
    """Show full details for a single coffee matching the query (name or URL)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = find_coffee(conn, query, "detail")
    if not row:
        conn.close()
        return

    print(f"\n{'━' * 60}")
    print(f"  {row['name']}")
    print(f"{'━' * 60}")
    print(
        f"  Score: {row['total_score']}  |  Price: ${row['price']}  |  {'In Stock ✅' if row['in_stock'] else 'Out of Stock ❌'}"
    )
    print(f"  URL: {row['url']}")
    print("\n  ── Cupping Scores ──")
    for label, key in [
        ("Dry Fragrance", "dry_fragrance"),
        ("Wet Aroma", "wet_aroma"),
        ("Brightness", "brightness"),
        ("Flavor", "flavor"),
        ("Body", "body"),
        ("Finish", "finish"),
        ("Sweetness", "sweetness"),
        ("Clean Cup", "clean_cup"),
        ("Complexity", "complexity"),
        ("Uniformity", "uniformity"),
    ]:
        v = row[key]
        if v is not None:
            bar = "█" * int(v) + "░" * (10 - int(v))
            print(f"    {label:14s} {bar} {v}")

    print("\n  ── Flavor Profile ──")
    flavs = [
        (label, row[k])
        for label, k in [
            ("Floral", "fl_floral"),
            ("Honey", "fl_honey"),
            ("Sugars", "fl_sugars"),
            ("Caramel", "fl_caramel"),
            ("Fruits", "fl_fruits"),
            ("Citrus", "fl_citrus"),
            ("Berry", "fl_berry"),
            ("Cocoa", "fl_cocoa"),
            ("Nuts", "fl_nuts"),
            ("Rustic", "fl_rustic"),
            ("Spice", "fl_spice"),
            ("Body", "fl_body"),
        ]
        if row[k] is not None
    ]
    for name, v in sorted(flavs, key=lambda x: -x[1]):
        bar = "█" * int(v) + "░" * (5 - int(v))
        print(f"    {name:10s} {bar} {v}")

    print("\n  ── Specs ──")
    for label, key in [
        ("Processing", "processing"),
        ("Cultivar", "cultivar"),
        ("Grade", "grade"),
        ("Appearance", "appearance"),
        ("Type", "type"),
    ]:
        if row[key]:
            print(f"    {label:14s} {row[key]}")
    if row["espresso_recommended"]:
        print(f"    {'Espresso':14s} ☕ Recommended")
    if row["roast_recommendations"]:
        print(f"    {'Roast':14s} {row['roast_recommendations']}")

    if row["overview"]:
        print("\n  ── Overview ──")
        print(f"    {row['overview']}")

    if row["cupping_notes"]:
        print("\n  ── Cupping Notes ──")
        # Word wrap at ~70 chars
        words = row["cupping_notes"].split()
        line = "    "
        for w in words:
            if len(line) + len(w) > 74:
                print(line)
                line = "    "
            line += w + " "
        if line.strip():
            print(line)

    print()
    conn.close()


def summary(no_decaf=False):
    """Print a formatted summary of all coffees in the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM coffees ORDER BY total_score DESC").fetchall()
    if no_decaf:
        rows = [r for r in rows if not _is_decaf(r)]
    if not rows:
        print("Database is empty. Run 'all' or 'single' first.")
        return

    in_stock = [r for r in rows if r["in_stock"]]
    avg_score = sum(r["total_score"] for r in rows if r["total_score"]) / len(
        [r for r in rows if r["total_score"]]
    )
    avg_price = sum(r["price"] for r in rows if r["price"]) / len(
        [r for r in rows if r["price"]]
    )

    print(f"\n{'━' * 70}")
    print("  SWEET MARIA'S GREEN COFFEE — DATABASE SUMMARY")
    print(f"  {len(rows)} coffees ({len(in_stock)} in stock)")
    print(f"  Avg Score: {avg_score:.1f} | Avg Price: ${avg_price:.2f}")
    print(f"{'━' * 70}\n")

    # Group by origin/processing
    by_process = {}
    for r in rows:
        p = r["processing"] or "Unknown"
        by_process.setdefault(p, []).append(r)

    print(
        f"  By Processing: {
            ', '.join(
                f'{k}({len(v)})'
                for k, v in sorted(by_process.items(), key=lambda x: -len(x[1]))
            )
        }\n"
    )

    # Per-coffee summary
    for i, r in enumerate(rows, 1):
        stock = "✅" if r["in_stock"] else "❌"
        espresso = " ☕" if r["espresso_recommended"] else ""
        print(f"  {i:2d}. {stock} {r['name']}")
        print(f"      Score: {r['total_score']}  Price: ${r['price']}{espresso}")

        # Cupping scores bar
        dims = [
            ("Frag", r["dry_fragrance"]),
            ("Aroma", r["wet_aroma"]),
            ("Bright", r["brightness"]),
            ("Flavor", r["flavor"]),
            ("Body", r["body"]),
            ("Finish", r["finish"]),
            ("Sweet", r["sweetness"]),
            ("Clean", r["clean_cup"]),
            ("Cmplx", r["complexity"]),
            ("Unif", r["uniformity"]),
        ]
        scored = [(n, v) for n, v in dims if v is not None]
        if scored:
            bars = " ".join(f"{n}:{v}" for n, v in scored)
            print(f"      [{bars}]")

        # Top flavors
        flavs = [
            ("Cocoa", r["fl_cocoa"]),
            ("Fruits", r["fl_fruits"]),
            ("Caramel", r["fl_caramel"]),
            ("Berry", r["fl_berry"]),
            ("Citrus", r["fl_citrus"]),
            ("Nuts", r["fl_nuts"]),
            ("Spice", r["fl_spice"]),
            ("Sugars", r["fl_sugars"]),
            ("Floral", r["fl_floral"]),
            ("Honey", r["fl_honey"]),
            ("Rustic", r["fl_rustic"]),
            ("Body", r["fl_body"]),
        ]
        top_flavs = sorted(
            [(n, v) for n, v in flavs if v and v > 0], key=lambda x: -x[1]
        )[:5]
        if top_flavs:
            print(f"      Flavors: {', '.join(f'{n}({v})' for n, v in top_flavs)}")

        # Specs one-liner
        specs = []
        if r["processing"]:
            specs.append(r["processing"])
        if r["cultivar"]:
            specs.append(r["cultivar"])
        if r["roast_recommendations"]:
            # Abbreviate roast rec
            roast = r["roast_recommendations"]
            if len(roast) > 40:
                roast = roast[:40] + "…"
            specs.append(f"Roast: {roast}")
        if specs:
            print(f"      {' | '.join(specs)}")
        print()

    conn.close()


def tried(url: str, rating: str = "0", notes: str = None):
    """Mark a coffee as tried. Rating: +, 0, or -"""
    if rating not in ("+", "0", "-"):
        print(f"Invalid rating '{rating}'. Use +, 0, or -")
        return
    if not url.startswith("https://www.sweetmarias.com/"):
        print(f"⚠️  URL doesn't look like a Sweet Maria's product: {url}")
    conn = init_db()
    conn.row_factory = sqlite3.Row
    coffee = conn.execute("SELECT name FROM coffees WHERE url = ?", (url,)).fetchone()
    conn.execute(
        "INSERT OR REPLACE INTO tried (url, rating, notes, tried_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        (url, rating, notes),
    )
    conn.commit()
    symbol = {"+": "👍", "0": "😐", "-": "👎"}[rating]
    name = coffee["name"] if coffee else url.split("/")[-1].replace(".html", "")
    print(f"  {symbol} Marked as tried: {name} [{rating}]")
    if not coffee:
        print("     (no flavor data yet — will be scraped when you run recommend)")
    if notes:
        print(f"     Notes: {notes}")
    conn.close()


def tried_list():
    """Show all coffees marked as tried."""
    conn = init_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT t.url, t.rating, t.notes, t.tried_at, c.name, c.total_score
        FROM tried t LEFT JOIN coffees c ON t.url = c.url
        ORDER BY t.tried_at DESC"""
    ).fetchall()
    if not rows:
        print("\nNo coffees marked as tried yet.")
        print("  Usage: python3 scrape_sm.py tried <url> [+|0|-] [notes]")
        conn.close()
        return
    print(f"\n  Tried coffees ({len(rows)}):\n")
    for r in rows:
        name = r["name"] or r["url"].split("/")[-1].replace(".html", "")
        print(f"  {r['rating']:1s}  {name}")
        print(f"     {r['url']}")
        if r["notes"]:
            print(f"     {r['notes']}")
        if r["notes"]:
            print(f"      Notes: {r['notes']}")
    print()
    conn.close()


if __name__ == "__main__":
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--verbose", action="store_true", help="show debug logging")
    common.add_argument("--quiet", action="store_true", help="suppress info logging")
    common.add_argument(
        "--decaf",
        action="store_true",
        help="include decaf coffees (excluded by default)",
    )

    parser = argparse.ArgumentParser(
        prog="scrape_sm.py",
        description="Sweet Maria's Green Coffee Scraper & Data Management",
        epilog=f"Database: {DB_PATH}",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- single ---
    p = sub.add_parser("single", parents=[common], help="scrape one product page")
    p.add_argument("url", help="product URL to scrape")

    # --- all ---
    p = sub.add_parser("all", parents=[common], help="scrape all green coffees")
    p.add_argument(
        "--include-oos",
        action="store_true",
        help="also scrape out-of-stock coffees",
    )

    # --- show ---
    sub.add_parser("show", parents=[common], help="show compact list of all coffees")

    # --- stock ---
    sub.add_parser("stock", parents=[common], help="show weekly stock velocity")

    # --- summary ---
    sub.add_parser(
        "summary", parents=[common], help="show formatted summary of all coffees"
    )

    # --- detail ---
    p = sub.add_parser(
        "detail",
        parents=[common],
        help="show full details for a coffee (fuzzy name or URL)",
    )
    p.add_argument("query", help="coffee name (fuzzy) or URL")

    # --- tried ---
    p = sub.add_parser(
        "tried",
        parents=[common],
        help="mark a coffee as tried, or list tried coffees",
    )
    p.add_argument("url", nargs="?", help="product URL")
    p.add_argument(
        "rating",
        nargs="?",
        default="0",
        choices=["+", "0", "-"],
        help="rating: + (loved), 0 (neutral), - (disliked) (default: %(default)s)",
    )
    p.add_argument("notes", nargs="?", help="tasting notes")
    p.add_argument("--list", action="store_true", help="show all tried coffees")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s: %(message)s", stream=sys.stderr
    )

    no_decaf = not args.decaf

    if args.command == "single":
        main_single(args.url)
    elif args.command == "all":
        main_all(include_out_of_stock=args.include_oos)
    elif args.command == "show":
        show(no_decaf=no_decaf)
    elif args.command == "stock":
        stock()
    elif args.command == "summary":
        summary(no_decaf=no_decaf)
    elif args.command == "detail":
        detail(args.query)
    elif args.command == "tried":
        if args.list or not args.url:
            tried_list()
        else:
            tried(args.url, args.rating, args.notes)
