#!/usr/bin/env python3
"""Coffee flavor space analysis — recommendations, clustering, exploration.

Uses the database populated by scrape_sm.py. Run with --help for usage.
"""

import argparse
import logging
import random
import sqlite3
import sys
import time
from datetime import date

from embed import build_embeddings, load_embeddings
from scrape_sm import (
    ALL_COLS,
    CUPPING_COLS,
    DB_PATH,
    DELAY,
    DIMS,
    DIM_NAMES,
    SCALES,
    find_coffee,
    init_db,
    save_coffee,
    scrape_product,
    _is_blend,
    _is_decaf,
    _matches_exclude,
)

logger = logging.getLogger(__name__)

# --- Design principles for recommendations ---
# 1. Any coffee surfaced as a recommendation MUST be in stock.
#    Analysis may run across all coffees, but display picks (purest expression,
#    exemplars, contrast pairs, "to explore" suggestions) must be purchasable.
# 2. Every recommendation should include a contrast pair: a second in-stock coffee
#    that is maximally different, so the user can triangulate the flavor space by
#    tasting both ends.

# --- Normalization state ---

_NORM_STATS = None


def compute_pop_stats(rows):
    """Compute per-dimension population mean and stddev from scored coffees."""
    means = []
    stddevs = []
    for c in ALL_COLS:
        vals = [r[c] for r in rows if r[c] is not None]
        if not vals:
            means.append(0)
            stddevs.append(1)
            continue
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        means.append(m)
        stddevs.append(sd if sd > 0 else 0.01)
    return means, stddevs


def init_zscore(conn):
    """Compute and set population stats for z-score normalization."""
    global _NORM_STATS
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM coffees WHERE dry_fragrance IS NOT NULL AND total_score IS NOT NULL"
    ).fetchall()
    if rows:
        _NORM_STATS = compute_pop_stats(rows)
        logger.info("zscore normalization enabled with %d coffees, clamp=±3", len(rows))


def to_vector(row):
    """Convert a coffee row to a 22-dimensional vector."""
    if _NORM_STATS is None:
        return [(row[c] or 0) / s for c, s in zip(ALL_COLS, SCALES)]
    means, stddevs = _NORM_STATS
    vec = []
    for i, c in enumerate(ALL_COLS):
        val = row[c] or 0
        z = (val - means[i]) / stddevs[i] if stddevs[i] > 0 else 0
        vec.append(max(-3.0, min(3.0, z)))
    return vec


def _dscale():
    """Display scale: SCALES in old mode, [1]*22 in zscore mode (values already in σ)."""
    if _NORM_STATS is None:
        return SCALES
    return [1] * 22


# --- Distance functions ---

_DISTANCE_METRIC = "l2"
_TEXT_WEIGHT = 0.0
_EMBEDDINGS = None


def ensure_embeddings(conn):
    """Build or update text embeddings if any coffees with notes are missing them."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS embeddings (
        url TEXT PRIMARY KEY,
        model TEXT NOT NULL,
        vector BLOB NOT NULL,
        text_hash TEXT NOT NULL)"""
    )
    missing = conn.execute(
        """SELECT COUNT(*) FROM coffees
        WHERE cupping_notes IS NOT NULL
        AND url NOT IN (SELECT url FROM embeddings)"""
    ).fetchone()[0]
    if missing == 0:
        return
    logger.info("embeddings stale: %d coffees need embedding, building now", missing)
    n_built = build_embeddings(conn)
    logger.info("embeddings updated: %d new embeddings computed", n_built)


def _init_text_embeddings(conn):
    """Load text embeddings if text weight is active."""
    global _EMBEDDINGS
    if _TEXT_WEIGHT <= 0:
        return
    _EMBEDDINGS = load_embeddings(conn)
    if not _EMBEDDINGS:
        logger.warning(
            "text weight set but no embeddings found, run: python embed.py build"
        )
    else:
        logger.info(
            "loaded %d text embeddings (text_weight=%.2f)",
            len(_EMBEDDINGS),
            _TEXT_WEIGHT,
        )


def compute_dim_weights(all_vecs):
    """Compute variance-based dimension weights from a set of vectors."""
    dim_variances = []
    for i in range(22):
        vals = [v[i] for v in all_vecs]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        dim_variances.append(var if var > 0 else 0.001)
    total_var = sum(dim_variances)
    return [v / total_var * 22 for v in dim_variances]


def compute_dim_stddevs(all_vecs):
    """Compute per-dimension standard deviations from a set of normalized vectors."""
    n = len(all_vecs)
    stddevs = []
    for i in range(22):
        vals = [v[i] for v in all_vecs]
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        stddevs.append(var**0.5 if var > 0 else 0.01)
    return stddevs


def make_distance_fn(all_vecs):
    """Return a distance function based on the selected metric and the population vectors.

    The returned function has signature: distance(a, b, url_a=None, url_b=None).
    When both URLs are provided, text embeddings are available, and _TEXT_WEIGHT > 0,
    the function blends numeric distance with text cosine distance.
    """
    metric = _DISTANCE_METRIC

    if metric == "l1":

        def numeric_distance(a, b):
            return sum(abs(x - y) for x, y in zip(a, b))

    elif metric == "cosine":

        def numeric_distance(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = sum(x * x for x in a) ** 0.5
            mag_b = sum(x * x for x in b) ** 0.5
            if mag_a == 0 or mag_b == 0:
                return 1.0
            return 1.0 - dot / (mag_a * mag_b)

    elif metric == "mahalanobis":
        import numpy as np

        arr = np.array(all_vecs)
        cov = np.cov(arr, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        cov_inv = np.linalg.inv(cov)

        def numeric_distance(a, b):
            diff = np.array(a) - np.array(b)
            return float(np.sqrt(diff @ cov_inv @ diff))

    else:  # l2 (default)
        weights = compute_dim_weights(all_vecs)

        def numeric_distance(a, b):
            return sum(w * (x - y) ** 2 for x, y, w in zip(a, b, weights)) ** 0.5

    # Wrap with text blending
    text_weight = _TEXT_WEIGHT
    embeddings = _EMBEDDINGS

    if text_weight <= 0 or not embeddings:

        def distance(a, b, url_a=None, url_b=None):
            return numeric_distance(a, b)

    else:

        def distance(a, b, url_a=None, url_b=None):
            nd = numeric_distance(a, b)
            if url_a is None or url_b is None:
                return nd
            emb_a = embeddings.get(url_a)
            emb_b = embeddings.get(url_b)
            if emb_a is None or emb_b is None:
                return nd
            # Cosine distance: embeddings are normalized, so 1 - dot = cosine dist
            text_dist = 1.0 - float(emb_a @ emb_b)
            return (1.0 - text_weight) * nd + text_weight * text_dist

    return distance


def make_weighted_distance(all_vecs):
    """Return a distance function based on the selected metric."""
    return make_distance_fn(all_vecs)


# --- Clustering ---


def pick_k_elbow(vecs_list, distance_fn, k_min=2, k_max=9, seed=42):
    """Pick k using silhouette score (best cluster separation/cohesion).

    Tests k_min through k_max, selects the k with highest average silhouette (k>=3).
    Also returns k=2 results for coarse view and metadata.
    """
    rng = random.Random(seed)
    dims = len(vecs_list[0])

    def _kmeans(vecs, k):
        centers = [list(vecs[i]) for i in rng.sample(range(len(vecs)), k)]
        prev_labels = None
        for _ in range(100):
            labels = [
                min(range(k), key=lambda ci: distance_fn(v, centers[ci])) for v in vecs
            ]
            if labels == prev_labels:
                break
            prev_labels = labels
            for ci in range(k):
                members = [vecs[j] for j in range(len(vecs)) if labels[j] == ci]
                if members:
                    centers[ci] = [
                        sum(v[d] for v in members) / len(members) for d in range(dims)
                    ]
        return labels, centers

    def _inertia(vecs, labels, centers):
        return sum(
            distance_fn(vecs[i], centers[labels[i]]) ** 2 for i in range(len(vecs))
        )

    def _silhouette(vecs, labels, k):
        """Average silhouette score: how well each point fits its cluster vs nearest other."""
        n = len(vecs)
        scores = []
        for i in range(n):
            ci = labels[i]
            same = [j for j in range(n) if labels[j] == ci and j != i]
            if not same:
                scores.append(0)
                continue
            a = sum(distance_fn(vecs[i], vecs[j]) for j in same) / len(same)
            b = float("inf")
            for ck in range(k):
                if ck == ci:
                    continue
                others = [j for j in range(n) if labels[j] == ck]
                if others:
                    mean_d = sum(distance_fn(vecs[i], vecs[j]) for j in others) / len(
                        others
                    )
                    b = min(b, mean_d)
            scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
        return sum(scores) / len(scores)

    # Compute clustering and scores for each k
    results = {}
    silhouettes = {}
    for k in range(k_min, k_max + 1):
        labels, centers = _kmeans(vecs_list, k)
        results[k] = (labels, centers, _inertia(vecs_list, labels, centers))
        silhouettes[k] = _silhouette(vecs_list, labels, k)

    # Pick fine k with highest silhouette score (k>=3, since k=2 is the coarse view)
    fine_min = max(k_min, 3)
    fine_k = max(range(fine_min, k_max + 1), key=lambda k: silhouettes[k])

    # Also compute elbow for metadata
    ks = list(range(k_min, k_max + 1))
    inertias = [results[k][2] for k in ks]
    drops = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    accels = [drops[i] - drops[i + 1] for i in range(len(drops) - 1)]
    best_idx = accels.index(max(accels)) + 1
    elbow_k = ks[best_idx]

    return {
        "coarse": {"k": 2, "labels": results[2][0], "centers": results[2][1]},
        "fine": {
            "k": fine_k,
            "labels": results[fine_k][0],
            "centers": results[fine_k][1],
        },
        "meta": {
            "k_min": k_min,
            "k_max": k_max,
            "elbow_k": elbow_k,
            "fine_k": fine_k,
            "method": "silhouette",
            "silhouettes": silhouettes,
            "inertias": {k: results[k][2] for k in ks},
        },
    }


def hopkins_statistic(vecs_list, distance_fn, sample_frac=0.3, seed=42):
    """Compute Hopkins statistic to test for clustering tendency.

    Returns a value in [0, 1]:
      ~0.5 = uniformly distributed (no clusters)
      ~1.0 = highly clustered
    """
    rng = random.Random(seed)
    n = len(vecs_list)
    dims = len(vecs_list[0])
    m = max(1, int(n * sample_frac))

    mins = [min(v[d] for v in vecs_list) for d in range(dims)]
    maxs = [max(v[d] for v in vecs_list) for d in range(dims)]

    sample_indices = rng.sample(range(n), m)

    w_dists = []
    for i in sample_indices:
        nn_dist = min(
            distance_fn(vecs_list[i], vecs_list[j]) for j in range(n) if j != i
        )
        w_dists.append(nn_dist)

    u_dists = []
    for _ in range(m):
        rand_point = [rng.uniform(mins[d], maxs[d]) for d in range(dims)]
        nn_dist = min(distance_fn(rand_point, vecs_list[j]) for j in range(n))
        u_dists.append(nn_dist)

    sum_u = sum(u_dists)
    sum_w = sum(w_dists)
    if sum_u + sum_w == 0:
        return 0.5
    return sum_u / (sum_u + sum_w)


def cluster_label(center, centroid, stddevs):
    """Describe a cluster by its deviation from the centroid (>0.5 stddev)."""
    ds = _dscale()
    diffs = [
        (DIM_NAMES[i], center[i] * ds[i] - centroid[i] * ds[i])
        for i in range(22)
        if abs(center[i] - centroid[i]) > 0.5 * stddevs[i]
    ]
    diffs.sort(key=lambda x: -abs(x[1]))
    sfx = "σ" if _NORM_STATS else ""
    high = [f"{n}+{d:.1f}{sfx}" for n, d in diffs[:2] if d > 0]
    low = [f"{n}{d:.1f}{sfx}" for n, d in diffs[:1] if d < 0]
    return ", ".join(high + low) or "balanced"


# --- Utility ---


def _ordinal(n):
    """Return number with ordinal suffix: 1st, 2nd, 3rd, 4th, ..."""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd'][n % 10] if n % 10 < 4 else 'th'}"


def pearson_r(xs, ys):
    """Compute Pearson correlation coefficient between two lists."""
    n = len(xs)
    if n < 3:
        return 0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0


def get_scored_coffees(conn, no_decaf=False, exclude_urls=None, available_only=False):
    """Get all non-blend coffees with flavor data."""
    rows = conn.execute(
        """SELECT * FROM coffees
        WHERE dry_fragrance IS NOT NULL AND total_score IS NOT NULL"""
    ).fetchall()
    rows = [r for r in rows if not _is_blend(r)]
    if no_decaf:
        rows = [r for r in rows if not _is_decaf(r)]
    if exclude_urls:
        rows = [r for r in rows if not _matches_exclude(r, exclude_urls)]
    if available_only:
        rows = [r for r in rows if r["in_stock"]]
    return rows


# --- Commands ---


def _scrape_missing(urls):
    """Scrape a list of URLs that are in tried but not in coffees."""
    conn = init_db()
    for url in urls:
        print(f"  Scraping: {url.split('/')[-1]}")
        try:
            data = scrape_product(url)
        except Exception as e:
            logger.warning("scrape missing failed for %s: %s", url, e)
            data = None
        if data and data.get("name"):
            save_coffee(conn, data)
            print(f"    ✅ {data['name']} — Score: {data['total_score']}")
        else:
            print("    ⚠️  Could not scrape (page may be gone) — marking as unavailable")
            conn.execute("INSERT OR IGNORE INTO coffees (url) VALUES (?)", (url,))
            conn.commit()
        time.sleep(DELAY)
    conn.close()


def recommend(no_decaf=False, exclude_urls=None, offline=False, top_n=3):
    """Recommend coffees: cluster map, dimension leaders, max variation from tried."""
    random.seed(42)

    conn = init_db()
    conn.row_factory = sqlite3.Row

    # Auto-scrape missing tried coffees
    missing = conn.execute(
        """SELECT t.url FROM tried t
        LEFT JOIN coffees c ON t.url = c.url
        WHERE c.url IS NULL"""
    ).fetchall()
    if missing and not offline:
        print(
            f"\n\U0001f50d Scraping {len(missing)} tried coffee(s) missing flavor data..."
        )
        conn.close()
        _scrape_missing([r["url"] for r in missing])
        conn = init_db()
        conn.row_factory = sqlite3.Row

    tried_rows = conn.execute(
        """SELECT c.*, t.rating FROM tried t
        JOIN coffees c ON t.url = c.url"""
    ).fetchall()
    if not tried_rows:
        print("\n⚠️  No tried coffees have flavor data. Mark some with:")
        print('  ./scrape_sm.py tried "<url>" [+|0|-] [notes]')
        conn.close()
        return

    untried = conn.execute(
        """SELECT c.* FROM coffees c
        WHERE c.in_stock = 1 AND c.url NOT IN (SELECT url FROM tried)
        AND c.total_score IS NOT NULL"""
    ).fetchall()
    if no_decaf:
        untried = [c for c in untried if not _is_decaf(c)]
    if exclude_urls:
        untried = [c for c in untried if not _matches_exclude(c, exclude_urls)]
    untried = [c for c in untried if not _is_blend(c)]

    if not untried:
        print("\nNo untried in-stock coffees in the database.")
        conn.close()
        return

    # --- Shared setup ---
    all_scored = get_scored_coffees(conn, no_decaf=no_decaf)
    all_vecs = [to_vector(c) for c in all_scored]
    weighted_distance = make_weighted_distance(all_vecs)
    stddevs = compute_dim_stddevs(all_vecs)

    # Tried vectors (exclude blends from distance calc)
    tried_for_distance = [r for r in tried_rows if not _is_blend(r)]
    if not tried_for_distance:
        tried_for_distance = tried_rows
    tried_vectors = [(to_vector(r), r["rating"], r["url"]) for r in tried_for_distance]
    rating_weights = {"+": 0.5, "0": 1.0, "-": 2.0}

    # --- K-means clustering ---
    km = pick_k_elbow(all_vecs, weighted_distance)
    best_k = km["fine"]["k"]
    labels = km["fine"]["labels"]
    centers = km["fine"]["centers"]

    # Map coffees to clusters
    tried_urls = set(r["url"] for r in tried_rows)
    centroid = [sum(v[d] for v in all_vecs) / len(all_vecs) for d in range(22)]
    clusters = []
    for ci in range(best_k):
        members = [all_scored[j] for j in range(len(all_scored)) if labels[j] == ci]
        tried_in = [c for c in members if c["url"] in tried_urls]
        untried_in = [c for c in members if c["url"] not in tried_urls and c in untried]
        label = cluster_label(centers[ci], centroid, stddevs)
        clusters.append(
            {
                "label": label,
                "members": members,
                "tried": tried_in,
                "untried": untried_in,
                "center": centers[ci],
            }
        )

    def print_contrast(reference, ref_vec, pool, indent="      "):
        """Print the most contrasting coffee: nearest to the inverted vector."""
        inverted = [2 * centroid[i] - ref_vec[i] for i in range(22)]
        contrast = min(
            pool,
            key=lambda c: (
                weighted_distance(to_vector(c), inverted)
                if c["url"] != reference["url"]
                else float("inf")
            ),
        )
        cvec = to_vector(contrast)
        ds = _dscale()
        cdiffs = [
            (DIMS[j][0], cvec[j] * ds[j], cvec[j] * ds[j] - ref_vec[j] * ds[j])
            for j in range(22)
            if abs(cvec[j] - ref_vec[j]) > 0.5 * stddevs[j]
        ]
        cdiffs.sort(key=lambda x: -abs(x[2]))
        more = [f"{n} {v:.1f}" for n, v, d in cdiffs if d > 0][:3]
        less = [f"{n} {v:.1f}" for n, v, d in cdiffs if d < 0][:3]
        print(f"{indent}Contrast: {contrast['name']}")
        print(f"{indent}  {contrast['url']}")
        desc = []
        if more:
            desc.append(f"more {', '.join(more)}")
        if less:
            desc.append(f"less {', '.join(less)}")
        if desc:
            print(f"{indent}  {'; '.join(desc)}")

    # --- OUTPUT ---
    print(f"\n{'━' * 60}")
    print("  RECOMMENDATIONS")
    print(f"  ({len(tried_rows)} tried, {len(untried)} candidates, {best_k} clusters)")
    print(f"{'━' * 60}")

    # Section 1: Cluster map with exemplars for ALL clusters
    meta = km["meta"]
    elbow_k = meta["elbow_k"]
    silhouettes = meta["silhouettes"]
    sil_summary = "  ".join(f"k={k}:{silhouettes[k]:.3f}" for k in sorted(silhouettes))
    hopkins = hopkins_statistic(all_vecs, weighted_distance)
    if hopkins > 0.75:
        hopkins_desc = "strong clustering tendency"
    elif hopkins > 0.6:
        hopkins_desc = "moderate clustering tendency"
    elif hopkins > 0.5:
        hopkins_desc = "weak clustering tendency"
    else:
        hopkins_desc = "no clustering tendency (uniform)"
    print(f"\n  ── Cluster Map (k={best_k}) ──")
    print(f"  Hopkins statistic: {hopkins:.3f} — {hopkins_desc}")
    print(
        f"  Silhouette method: tested k={meta['k_min']}–{meta['k_max']}, "
        f"best silhouette at k={best_k} ({silhouettes[best_k]:.3f}), "
        f"elbow at k={elbow_k}"
    )
    print(f"  Silhouettes: {sil_summary}\n")
    for cl in sorted(clusters, key=lambda x: -len(x["untried"])):
        n_tried = len(cl["tried"])
        n_untried = len(cl["untried"])
        explored_mark = "" if n_tried > 0 else " ⚡ UNEXPLORED"
        print(
            f"    {cl['label']} ({len(cl['members'])} total, "
            f"{n_tried} tried, {n_untried} available){explored_mark}"
        )
        ds = _dscale()
        profile_diffs = [
            (DIM_NAMES[i], cl["center"][i] * ds[i] - centroid[i] * ds[i])
            for i in range(22)
        ]
        profile_diffs.sort(key=lambda x: -x[1])
        highs = [f"{n}+{d:.1f}" for n, d in profile_diffs[:3] if d > 0.3 * stddevs[0]]
        lows = [f"{n}{d:.1f}" for n, d in profile_diffs[-2:] if d < -0.3 * stddevs[0]]
        if highs or lows:
            print(f"      Character: {', '.join(highs + lows)}")
        if cl["tried"]:
            center_vec = cl["center"]
            member_dists = [
                weighted_distance(to_vector(m), center_vec) for m in cl["members"]
            ]
            member_dists.sort()
            for tc in cl["tried"]:
                tv = to_vector(tc)
                d = weighted_distance(tv, center_vec)
                pct = (
                    sum(1 for md in member_dists if md <= d) * 100 // len(member_dists)
                )
                if pct <= 33:
                    desc = "central"
                elif pct <= 66:
                    desc = "mid"
                else:
                    desc = "edge"
                rating = next(
                    (r["rating"] for r in tried_rows if r["url"] == tc["url"]), "?"
                )
                sym = {"+": "👍", "0": "😐", "-": "👎"}.get(rating, "?")
                print(
                    f"      You tried: {sym} {tc['name']} ({_ordinal(pct)} pctl — {desc})"
                )
        exemplar = None
        if cl["untried"]:
            exemplar = max(cl["untried"], key=lambda c: c["total_score"] or 0)
        elif cl["members"]:
            exemplar = max(cl["members"], key=lambda c: c["total_score"] or 0)
        if exemplar:
            stock = "✅" if exemplar["url"] in {c["url"] for c in untried} else "❌"
            print(
                f"      Exemplar: {exemplar['name']} ({exemplar['total_score']}) {stock}"
            )
            print(f"      {exemplar['url']}")
            if len(untried) >= 2:
                print_contrast(exemplar, to_vector(exemplar), untried)
        print()

    # Section 2: Dimension leaders (available untried coffees)
    print("  ── Dimension Leaders (available) ──\n")
    interesting_dims = [
        ("Brightness", "brightness"),
        ("Body", "body"),
        ("Complexity", "complexity"),
        ("Sweetness", "sweetness"),
        ("Floral", "fl_floral"),
        ("Honey", "fl_honey"),
        ("Caramel", "fl_caramel"),
        ("Fruits", "fl_fruits"),
        ("Citrus", "fl_citrus"),
        ("Berry", "fl_berry"),
        ("Cocoa", "fl_cocoa"),
        ("Nuts", "fl_nuts"),
        ("Rustic", "fl_rustic"),
        ("Spice", "fl_spice"),
    ]
    for dim_name, col in interesting_dims:
        candidates = [(c, c[col]) for c in untried if c[col] is not None and c[col] > 0]
        if not candidates:
            continue
        best = max(candidates, key=lambda x: x[1])
        best_coffee = best[0]
        print(f"    Most {dim_name:10s}: {best_coffee['name']} ({best[1]})")
        print(f"      {best_coffee['url']}")
        lowest = min(candidates, key=lambda x: x[1])
        if lowest[0]["url"] != best_coffee["url"]:
            contrast_coffee = lowest[0]
            print(f"      Contrast: {contrast_coffee['name']} ({lowest[1]})")
            print(f"        {contrast_coffee['url']}")
            bvec = to_vector(best_coffee)
            cvec = to_vector(contrast_coffee)
            ds = _dscale()
            cdiffs = [
                (DIM_NAMES[j], cvec[j] * ds[j], cvec[j] * ds[j] - bvec[j] * ds[j])
                for j in range(22)
                if abs(cvec[j] - bvec[j]) > 0.5 * stddevs[j]
            ]
            cdiffs.sort(key=lambda x: -abs(x[2]))
            more = [f"{n} {v:.1f}" for n, v, d in cdiffs if d > 0][:3]
            less = [f"{n} {v:.1f}" for n, v, d in cdiffs if d < 0][:3]
            desc = []
            if more:
                desc.append(f"more {', '.join(more)}")
            if less:
                desc.append(f"less {', '.join(less)}")
            if desc:
                print(f"        {'; '.join(desc)}")
        print()
    print()

    # Section 3: Most different from what you know
    print(f"  ── Most Different From What You Know (top {top_n}) ──")
    print("  Maximum flavor distance from your tried coffees:\n")

    scored = []
    for coffee in untried:
        vec = to_vector(coffee)
        distances = []
        for tvec, rating, turl in tried_vectors:
            d = weighted_distance(vec, tvec, url_a=coffee["url"], url_b=turl)
            d *= rating_weights.get(rating, 1.0)
            distances.append(d)
        min_dist = min(distances)
        min_idx = distances.index(min(distances))
        closest = tried_for_distance[min_idx]
        cvec = tried_vectors[min_idx][0]
        scored.append((coffee, min_dist, vec, closest, cvec))

    scored.sort(key=lambda x: -x[1])
    ds = _dscale()

    for i, (c, dist, vec, closest, cvec) in enumerate(scored[:top_n], 1):
        closest_diffs = []
        for j, (name, _) in enumerate(DIMS):
            rv = vec[j] * ds[j]
            cv = cvec[j] * ds[j]
            d = rv - cv
            if abs(d) > 0.5:
                closest_diffs.append((name, rv, d))
        closest_diffs.sort(key=lambda x: -abs(x[2]))
        high = [f"{n} {v:.1f}" for n, v, d in closest_diffs if d > 0][:3]
        low = [f"{n} {v:.1f}" for n, v, d in closest_diffs if d < 0][:3]

        print(f"  {i:2d}. {c['name']}")
        print(f"      {c['url']}")
        if high:
            print(f"      Stands out for: {', '.join(high)}")
        if low:
            print(f"      Low in: {', '.join(low)}")
        print(f"      Most similar to tried: {closest['name']}")
        print_contrast(c, vec, untried)
        print()

    conn.close()


def insights(
    no_decaf=False, exclude_urls=None, clusters_only=False, available_only=False
):
    """Analyze the collection: outliers, clusters, superlatives, process patterns."""
    conn = init_db()
    conn.row_factory = sqlite3.Row
    rows = get_scored_coffees(
        conn,
        no_decaf=no_decaf,
        exclude_urls=exclude_urls,
        available_only=available_only,
    )

    vecs = [(r, to_vector(r)) for r in rows]
    all_vecs = [v for _, v in vecs]
    weighted_distance = make_weighted_distance(all_vecs)
    centroid = [sum(v[i] for v in all_vecs) / len(all_vecs) for i in range(22)]
    stddevs = compute_dim_stddevs(all_vecs)

    print(f"\n{'━' * 60}")
    print(f"  INSIGHTS — {len(rows)} coffees in collection")
    print(f"{'━' * 60}")

    if not clusters_only:
        # === SUPERLATIVES ===
        print("\n  ── Superlatives ──")
        interesting = [
            "fl_floral",
            "fl_honey",
            "fl_caramel",
            "fl_fruits",
            "fl_citrus",
            "fl_berry",
            "fl_cocoa",
            "fl_nuts",
            "fl_rustic",
            "fl_spice",
            "brightness",
            "body",
            "complexity",
        ]
        int_names = [
            "Floral",
            "Honey",
            "Caramel",
            "Fruits",
            "Citrus",
            "Berry",
            "Cocoa",
            "Nuts",
            "Rustic",
            "Spice",
            "Brightness",
            "Body",
            "Complexity",
        ]
        for name, col in zip(int_names, interesting):
            vals = [(r, r[col]) for r in rows if r[col] is not None]
            if not vals:
                continue
            top = max(vals, key=lambda x: x[1])
            if top[1] > 0:
                print(f"    Most {name:10s}: {top[0]['name']} ({top[1]})")

        # === OUTLIERS ===
        print("\n  ── Most Unusual (farthest from average) ──")
        dists_from_center = [(r, weighted_distance(v, centroid)) for r, v in vecs]
        dists_from_center.sort(key=lambda x: -x[1])
        for r, d in dists_from_center[:5]:
            v = to_vector(r)
            diffs = [
                (DIM_NAMES[i], v[i] * _dscale()[i] - centroid[i] * _dscale()[i])
                for i in range(22)
                if abs(v[i] - centroid[i]) > stddevs[i]
            ]
            diffs.sort(key=lambda x: -abs(x[1]))
            traits = ", ".join(
                f"{n}{'+' if d > 0 else ''}{d:.1f}" for n, d in diffs[:3]
            )
            print(f"    {r['name']} (dist={d:.2f})")
            print(f"      {traits}")

        print("\n  ── Most Typical (closest to average) ──")
        for r, d in dists_from_center[-3:]:
            print(f"    {r['name']} (dist={d:.2f})")

    # === K-MEANS CLUSTERING ===
    km = pick_k_elbow(all_vecs, weighted_distance)

    def _print_clusters(labels, centers, k, header):
        cluster_groups = [[] for _ in range(k)]
        for idx, (r, v) in enumerate(vecs):
            cluster_groups[labels[idx]].append((r, v))
        print(f"\n  ── {header} (k={k}) ──")
        for ci, group in enumerate(cluster_groups):
            if not group:
                continue
            label = cluster_label(centers[ci], centroid, stddevs)
            print(f"\n    Cluster {ci + 1}: {label} ({len(group)} coffees)")
            for r, _ in sorted(group, key=lambda x: -(x[0]["total_score"] or 0))[:5]:
                print(f"      {r['name']}")
            if len(group) > 5:
                print(f"      ... and {len(group) - 5} more")

    _print_clusters(
        km["coarse"]["labels"],
        km["coarse"]["centers"],
        km["coarse"]["k"],
        "Broad Clusters",
    )
    _print_clusters(
        km["fine"]["labels"],
        km["fine"]["centers"],
        km["fine"]["k"],
        "Fine-Grained Clusters",
    )

    if clusters_only:
        print()
        conn.close()
        return

    # === PROCESS PATTERNS ===
    print("\n  ── Processing Method Signatures ──")
    by_process = {}
    for r in rows:
        key = r["processing"] or "Unknown"
        by_process.setdefault(key, []).append(r)

    for proc, coffees in sorted(by_process.items(), key=lambda x: -len(x[1])):
        avgs = {}
        for i, (name, col) in enumerate(zip(DIM_NAMES, ALL_COLS)):
            vals = [r[col] for r in coffees if r[col] is not None]
            if vals:
                avgs[name] = sum(vals) / len(vals) - centroid[i] * _dscale()[i]
        top = sorted(avgs.items(), key=lambda x: -x[1])[:3]
        bot = sorted(avgs.items(), key=lambda x: x[1])[:2]
        high_str = ", ".join(f"{n}+{v:.1f}" for n, v in top if v > 0.2)
        low_str = ", ".join(f"{n}{v:.1f}" for n, v in bot if v < -0.2)
        desc = "; ".join(filter(None, [high_str, low_str]))
        print(f"    {proc} ({len(coffees)}): {desc or 'balanced'}")

    # === CORRELATIONS ===
    print("\n  ── Key Correlations ──")
    pairs = []
    for i in range(22):
        for j in range(i + 1, 22):
            xs = [
                r[ALL_COLS[i]]
                for r in rows
                if r[ALL_COLS[i]] is not None and r[ALL_COLS[j]] is not None
            ]
            ys = [
                r[ALL_COLS[j]]
                for r in rows
                if r[ALL_COLS[i]] is not None and r[ALL_COLS[j]] is not None
            ]
            corr = pearson_r(xs, ys)
            pairs.append((DIM_NAMES[i], DIM_NAMES[j], corr))
    pairs.sort(key=lambda x: -abs(x[2]))
    neg = sorted([(a, b, r) for a, b, r in pairs if r < 0], key=lambda x: x[2])[:6]
    pos = sorted([(a, b, r) for a, b, r in pairs if r > 0], key=lambda x: -x[2])[:6]
    if neg:
        print("    Trade-offs (can't have both):")
        for a, b, r in neg:
            print(f"      {a} vs {b} (r={r:.2f})")
    if pos:
        print("    Move together:")
        for a, b, r in pos:
            print(f"      {a} + {b} (r=+{r:.2f})")

    print()
    conn.close()


def profile(query, no_decaf=False):
    """Profile a single coffee: dimensions, archetype mix, contrast pair, cluster placement."""
    random.seed(42)

    conn = init_db()
    conn.row_factory = sqlite3.Row

    target = find_coffee(conn, query, "profile")
    if not target:
        conn.close()
        return
    if not target["dry_fragrance"]:
        print(f"Found '{target['name']}' but it has no flavor data.")
        conn.close()
        return

    tried_rows = conn.execute(
        """SELECT c.*, t.rating FROM tried t
        JOIN coffees c ON t.url = c.url WHERE c.dry_fragrance IS NOT NULL"""
    ).fetchall()
    if not tried_rows:
        print(
            "No tried coffees with flavor data. Mark some with: "
            "./scrape_sm.py tried <url> [+|0|-]"
        )
        conn.close()
        return

    all_scored = get_scored_coffees(conn, no_decaf=no_decaf)
    all_vecs = [to_vector(c) for c in all_scored]
    weighted_distance = make_weighted_distance(all_vecs)

    km = pick_k_elbow(all_vecs, weighted_distance)
    best_k = km["fine"]["k"]
    labels = km["fine"]["labels"]
    centers = km["fine"]["centers"]

    target_vec = to_vector(target)
    target_idx = next(
        (i for i, c in enumerate(all_scored) if c["url"] == target["url"]), None
    )
    if target_idx is not None:
        target_cluster = labels[target_idx]
    else:
        target_cluster = min(
            range(best_k), key=lambda ci: weighted_distance(target_vec, centers[ci])
        )

    centroid = [sum(v[i] for v in all_vecs) / len(all_vecs) for i in range(22)]
    stddevs = compute_dim_stddevs(all_vecs)
    cluster_members = [
        all_scored[j] for j in range(len(all_scored)) if labels[j] == target_cluster
    ]
    cl_label = cluster_label(centers[target_cluster], centroid, stddevs)

    tried_urls = set(r["url"] for r in tried_rows)
    tried_in_cluster = [c for c in cluster_members if c["url"] in tried_urls]

    tried_dists = []
    for tr in tried_rows:
        if tr["url"] == target["url"]:
            continue
        tvec = to_vector(tr)
        d = weighted_distance(target_vec, tvec, url_a=target["url"], url_b=tr["url"])
        tried_dists.append((tr, d, tvec))
    tried_dists.sort(key=lambda x: x[1])

    stock_str = "In Stock ✅" if target["in_stock"] else "Out of Stock ❌"
    print(f"\n{'━' * 60}")
    print(f"  PROFILE: {target['name']}")
    print(f"{'━' * 60}")
    print(f"  Score: {target['total_score']}  |  ${target['price']}  |  {stock_str}")
    print(f"  Processing: {target['processing']}")

    top_flavs = sorted(
        [
            (DIMS[d][0], target_vec[d] * _dscale()[d])
            for d in range(10, 22)
            if target_vec[d] * _dscale()[d] > 0.5
        ],
        key=lambda x: -x[1],
    )[:5]
    if top_flavs:
        print(f"  Top flavors: {', '.join(f'{n}:{v:.1f}' for n, v in top_flavs)}")

    print(f"\n  ── Cluster Placement (k={best_k}) ──")
    print(f'  Cluster: "{cl_label}" ({len(cluster_members)} coffees)')
    if tried_in_cluster:
        print("  You've tried from this cluster:")
        for tc in tried_in_cluster:
            r = next((t for t in tried_rows if t["url"] == tc["url"]), None)
            rating = r["rating"] if r else "?"
            sym = {"+": "👍", "0": "😐", "-": "👎"}.get(rating, "?")
            print(f"    {sym} {tc['name']}")
    else:
        print("  ⚡ You have NOT tried anything from this cluster!")

    if tried_dists:
        print("\n  ── Distance to Your Tried Coffees ──")
        for tr, d, tvec in tried_dists:
            sym = {"+": "👍", "0": "😐", "-": "👎"}.get(tr["rating"], "?")
            key_diffs = [
                (DIMS[j][0], target_vec[j] * _dscale()[j] - tvec[j] * _dscale()[j])
                for j in range(22)
                if abs(target_vec[j] - tvec[j]) > 0.1
            ]
            key_diffs.sort(key=lambda x: -abs(x[1]))
            diff_str = ", ".join(
                f"{n}{'+' if v > 0 else ''}{v:.1f}" for n, v in key_diffs[:3]
            )
            print(f"    {sym} {tr['name']}  dist={d:.3f}")
            if diff_str:
                print(f"       vs tried: {diff_str}")

    candidates = [
        c
        for c in all_scored
        if c["in_stock"]
        and c["url"] not in tried_urls
        and c["url"] != target["url"]
        and not _is_blend(c)
    ]
    if candidates:
        sim_scored = [
            (
                c,
                weighted_distance(
                    to_vector(c), target_vec, url_a=c["url"], url_b=target["url"]
                ),
            )
            for c in candidates
        ]
        sim_scored.sort(key=lambda x: x[1])

        print("\n  ── Most Similar Available Coffees ──")
        for i, (c, dist) in enumerate(sim_scored[:3], 1):
            cvec = to_vector(c)
            diffs = [
                (DIMS[j][0], cvec[j] * _dscale()[j] - target_vec[j] * _dscale()[j])
                for j in range(22)
                if abs(cvec[j] - target_vec[j]) > 0.5 * stddevs[j]
            ]
            diffs.sort(key=lambda x: -abs(x[1]))
            higher = [f"{n}+{d:.1f}" for n, d in diffs if d > 0][:3]
            lower = [f"{n}{d:.1f}" for n, d in diffs if d < 0][:2]
            print(f"    {i}. {c['name']}  (dist={dist:.3f})")
            print(f"       {c['url']}")
            desc = []
            if higher:
                desc.append(f"more {', '.join(higher)}")
            if lower:
                desc.append(f"less {', '.join(lower)}")
            if desc:
                print(f"       vs target: {'; '.join(desc)}")
            else:
                print("       Very close flavor match!")
            print()

    # ── Nearest In-Stock Substitute (only if target is out of stock) ──
    if not target["in_stock"]:
        in_stock_others = [
            c
            for c in all_scored
            if c["in_stock"] and c["url"] != target["url"] and not _is_blend(c)
        ]
        if in_stock_others:
            sub_scored = [
                (
                    c,
                    weighted_distance(
                        to_vector(c), target_vec, url_a=c["url"], url_b=target["url"]
                    ),
                )
                for c in in_stock_others
            ]
            sub_scored.sort(key=lambda x: x[1])
            best_sub, best_sub_dist = sub_scored[0]
            bvec = to_vector(best_sub)
            diffs = [
                (DIMS[j][0], bvec[j] * _dscale()[j] - target_vec[j] * _dscale()[j])
                for j in range(22)
                if abs(bvec[j] - target_vec[j]) > 0.5 * stddevs[j]
            ]
            diffs.sort(key=lambda x: -abs(x[1]))
            higher = [f"{n}+{d:.1f}" for n, d in diffs if d > 0][:3]
            lower = [f"{n}{d:.1f}" for n, d in diffs if d < 0][:2]
            print("\n  ── Nearest In-Stock Substitute ──")
            print(f"    {best_sub['name']}  (dist={best_sub_dist:.3f})")
            print(f"    {best_sub['url']}")
            print(f"    Score: {best_sub['total_score']}  |  ${best_sub['price']}")
            desc = []
            if higher:
                desc.append(f"more {', '.join(higher)}")
            if lower:
                desc.append(f"less {', '.join(lower)}")
            if desc:
                print(f"    vs target: {'; '.join(desc)}")
            else:
                print("    Nearly identical flavor profile!")

    # ── Contrast Pair (Antipode) ──
    # Reflect target through population centroid to find its opposite
    ndims = len(target_vec)
    antipode = [2 * centroid[d] - target_vec[d] for d in range(ndims)]
    in_stock_pool = [
        (i, c)
        for i, c in enumerate(all_scored)
        if c["in_stock"] and c["url"] != target["url"] and not _is_blend(c)
    ]
    if in_stock_pool:
        anti_scored = [
            (c, weighted_distance(to_vector(c), antipode)) for _, c in in_stock_pool
        ]
        anti_scored.sort(key=lambda x: x[1])
        contrast_coffee, contrast_dist = anti_scored[0]
        cvec = to_vector(contrast_coffee)
        # Find the dimension with the largest gap
        dim_gaps = [
            (abs((target_vec[d] - cvec[d]) * _dscale()[d]), d) for d in range(ndims)
        ]
        dim_gaps.sort(key=lambda x: -x[0])
        top_contrast_dim = DIM_NAMES[dim_gaps[0][1]]
        target_val = target_vec[dim_gaps[0][1]] * _dscale()[dim_gaps[0][1]]
        contrast_val = cvec[dim_gaps[0][1]] * _dscale()[dim_gaps[0][1]]
        print("\n  ── Contrast Pair (Antipode) ──")
        print(f"    {contrast_coffee['name']}")
        print(f"    {contrast_coffee['url']}")
        print(
            f"    Key axis: {top_contrast_dim} "
            f"(target={target_val:.1f} vs contrast={contrast_val:.1f})"
        )
        # Show top 3 secondary differences
        secondary = [
            (DIM_NAMES[d], (target_vec[d] - cvec[d]) * _dscale()[d])
            for _, d in dim_gaps[1:4]
            if abs((target_vec[d] - cvec[d]) * _dscale()[d]) > 0.2
        ]
        if secondary:
            sec_str = ", ".join(
                f"{n} {'↑' if g > 0 else '↓'}{abs(g):.1f}" for n, g in secondary
            )
            print(f"    Also differs: {sec_str}")

    # ── Dimension Rankings ──
    # For top standout dimensions, show where this coffee ranks in the catalog
    ds = _dscale()
    scored_dims = [(d, target_vec[d] * ds[d], DIM_NAMES[d]) for d in range(ndims)]
    scored_dims.sort(key=lambda x: -x[1])
    top_dims = [(d, val, name) for d, val, name in scored_dims if val > 0.3][:5]
    if top_dims:
        print("\n  ── Dimension Rankings ──")
        for d, val, name in top_dims:
            # Rank among all scored coffees on this dimension
            all_vals = sorted(
                [to_vector(c)[d] * ds[d] for c in all_scored], reverse=True
            )
            rank = next(
                (i + 1 for i, v in enumerate(all_vals) if v <= val), len(all_vals)
            )
            total = len(all_vals)
            pctile = int((total - rank) / total * 100)
            print(
                f"    {name}: {val:.1f} — #{rank} of {total} "
                f"({_ordinal(pctile)} pctile)"
            )

    # ── Archetype Decomposition ──
    import numpy as np

    vecs_all = [to_vector(c) for c in all_scored]
    pca = pca_reduce(vecs_all, variance_threshold=0.80)
    scores_pca = np.array(pca["scores"])
    components = np.array(pca["components"])
    mean_pca = np.array(pca["mean"])

    n_arch = _pick_n_archetypes(scores_pca)
    if isinstance(n_arch, tuple):
        n_arch = n_arch[0]
    aa = archetypal_analysis(scores_pca, n_arch)
    alpha_mat = np.array(aa["alpha"])
    arch_22d = np.array(aa["archetypes"]) @ components + mean_pca

    # Name each archetype by its top deviating dimensions
    arch_names = []
    for ai in range(n_arch):
        arch_vec = arch_22d[ai]
        diffs = [(DIM_NAMES[j], (arch_vec[j] - mean_pca[j]) * ds[j]) for j in range(22)]
        diffs.sort(key=lambda x: -abs(x[1]))
        top_pos = [n for n, d in diffs if d > 0.3 * stddevs[0]][:2]
        arch_names.append("/".join(top_pos) if top_pos else "Balanced")

    # Find target's row in all_scored to get its alpha weights
    if target_idx is not None:
        target_alpha = alpha_mat[target_idx]
    else:
        # Target not in scored set — project into PCA and solve for alpha
        target_pca = (np.array(target_vec) - mean_pca) @ components.T
        # Approximate: find nearest alpha by projecting
        dists_to_target = np.linalg.norm(scores_pca - target_pca, axis=1)
        nearest_idx = int(np.argmin(dists_to_target))
        target_alpha = alpha_mat[nearest_idx]

    parts = sorted(
        [(arch_names[a], target_alpha[a]) for a in range(n_arch)],
        key=lambda x: -x[1],
    )
    parts_str = " + ".join(f"{n} {w:.0%}" for n, w in parts if w > 0.05)
    print("\n  ── Archetype Decomposition ──")
    print(f"    {parts_str}")

    print()
    conn.close()


def _min_weight_matching(n, weight_fn):
    """Find the minimum-weight perfect matching for n items (n must be even).

    Uses recursive enumeration — fine for n <= ~14 (double-factorial growth).
    Returns list of (i, j) pairs.
    """
    items = list(range(n))

    def solve(remaining):
        if len(remaining) < 2:
            return [], 0
        first = remaining[0]
        rest = remaining[1:]
        best_pairs = None
        best_cost = float("inf")
        for idx, partner in enumerate(rest):
            leftover = rest[:idx] + rest[idx + 1:]
            sub_pairs, sub_cost = solve(leftover)
            cost = weight_fn(first, partner) + sub_cost
            if cost < best_cost:
                best_cost = cost
                best_pairs = [(first, partner)] + sub_pairs
        return best_pairs, best_cost

    pairs, _ = solve(items)
    return pairs


def pairs(queries):
    """Assign coffees into tasting contrast pairs based on flavor similarity.

    Uses the antipode method: for each coffee, reflect it through the population
    centroid to find its "opposite", then check which of the other input coffees
    is closest to that opposite. Mutual antipode matches are locked in first;
    remaining coffees are paired via minimum-weight matching on antipode distance.
    """
    conn = init_db()
    conn.row_factory = sqlite3.Row

    coffees = []
    for q in queries:
        c = find_coffee(conn, q, "pairs")
        if not c:
            conn.close()
            return
        if not c["dry_fragrance"]:
            print(f"Found '{c['name']}' but it has no flavor data.")
            conn.close()
            return
        coffees.append(c)

    if len(coffees) < 2:
        print("Need at least 2 coffees to form pairs.")
        conn.close()
        return

    all_scored = get_scored_coffees(conn, no_decaf=False)
    all_vecs = [to_vector(c) for c in all_scored]
    weighted_distance = make_weighted_distance(all_vecs)
    stddevs = compute_dim_stddevs(all_vecs)

    vecs = [to_vector(c) for c in coffees]
    n = len(coffees)

    # Compute population centroid.
    ndims = len(vecs[0])
    centroid = [sum(v[d] for v in all_vecs) / len(all_vecs) for d in range(ndims)]

    # For odd count, drop the coffee whose antipode is furthest from all others
    # in the set (hardest to pair as a contrast).
    leftover = None
    if n % 2 == 1:
        antipodes = [
            [2 * centroid[d] - vecs[i][d] for d in range(ndims)] for i in range(n)
        ]
        min_anti_dists = []
        for i in range(n):
            d = min(
                weighted_distance(antipodes[i], vecs[j]) for j in range(n) if j != i
            )
            min_anti_dists.append(d)
        drop_idx = max(range(n), key=lambda i: min_anti_dists[i])
        leftover = coffees[drop_idx]
        coffees = [c for i, c in enumerate(coffees) if i != drop_idx]
        vecs = [v for i, v in enumerate(vecs) if i != drop_idx]
        n -= 1

    # Compute antipode for each coffee (reflection through centroid).
    antipodes = [[2 * centroid[d] - vecs[i][d] for d in range(ndims)] for i in range(n)]

    # For each coffee, rank the others by proximity to its antipode.
    # antipode_pick[i] = index of the coffee closest to i's antipode.
    antipode_pick = []
    for i in range(n):
        ranked = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: weighted_distance(antipodes[i], vecs[j]),
        )
        antipode_pick.append(ranked[0])

    # Find mutual antipode matches (i picks j AND j picks i).
    paired = [False] * n
    matching = []
    for i in range(n):
        if paired[i]:
            continue
        j = antipode_pick[i]
        if not paired[j] and antipode_pick[j] == i:
            matching.append((i, j))
            paired[i] = True
            paired[j] = True

    # For remaining unpaired coffees, use min-weight matching on antipode distance.
    remaining = [i for i in range(n) if not paired[i]]
    if len(remaining) >= 2:
        # Weight = distance from j to antipode of i + distance from i to antipode of j
        # (symmetric contrast quality)
        def antipode_weight(ri, rj):
            i, j = remaining[ri], remaining[rj]
            return weighted_distance(antipodes[i], vecs[j]) + weighted_distance(
                antipodes[j], vecs[i]
            )

        sub_matching = _min_weight_matching(len(remaining), antipode_weight)
        for ri, rj in sub_matching:
            matching.append((remaining[ri], remaining[rj]))

    ds = _dscale()

    print(f"\n{'━' * 60}")
    print("  TASTING PAIRS — Contrast Assignments")
    print(f"  {n} coffees → {len(matching)} pairs")
    print(f"{'━' * 60}\n")

    for pair_num, (i, j) in enumerate(matching, 1):
        a, b = coffees[i], coffees[j]
        va, vb = vecs[i], vecs[j]

        # Determine pairing method.
        mutual = antipode_pick[i] == j and antipode_pick[j] == i
        method = "mutual antipode" if mutual else "antipode matching"

        # Find the dimension with the largest absolute gap.
        dim_gaps = []
        for d in range(ndims):
            gap = (va[d] - vb[d]) * ds[d]
            dim_gaps.append((abs(gap), gap, d))
        dim_gaps.sort(key=lambda x: -x[0])

        contrast_dim_idx = dim_gaps[0][2]
        contrast_gap = dim_gaps[0][1]
        contrast_name = DIM_NAMES[contrast_dim_idx]

        # Determine HIGH and LOW.
        if contrast_gap > 0:
            high, low = a, b
            high_val = va[contrast_dim_idx] * ds[contrast_dim_idx]
            low_val = vb[contrast_dim_idx] * ds[contrast_dim_idx]
        else:
            high, low = b, a
            high_val = vb[contrast_dim_idx] * ds[contrast_dim_idx]
            low_val = va[contrast_dim_idx] * ds[contrast_dim_idx]

        # Residual distance (all dims except contrast dim).
        residual = (
            sum((va[d] - vb[d]) ** 2 for d in range(ndims) if d != contrast_dim_idx)
            ** 0.5
        )

        dist = weighted_distance(
            va, vb, url_a=coffees[i]["url"], url_b=coffees[j]["url"]
        )

        print(f"  Pair {pair_num}: isolates {contrast_name}  [{method}]")
        print(f"  {'─' * 50}")
        print(f"    HIGH ({contrast_name}={high_val:.1f}): {high['name']}")
        print(f"    LOW  ({contrast_name}={low_val:.1f}): {low['name']}")
        print(
            f"    Gap: {abs(contrast_gap):.2f}  |  Residual dist: {residual:.3f}  |  Total dist: {dist:.3f}"
        )

        # Show secondary differences if any are notable.
        secondary = [
            (DIM_NAMES[d], gap * (1 if contrast_gap > 0 else -1))
            for _, gap, d in dim_gaps[1:4]
            if dim_gaps[0][0] > 0 and abs(gap) > 0.3 * stddevs[d] * ds[d]
        ]
        if secondary:
            sec_str = ", ".join(
                f"{n} {'↑' if g > 0 else '↓'}{abs(g):.1f}" for n, g in secondary
            )
            print(f"    Also differs: {sec_str}")
        print()

    if leftover:
        print(f"  ⚠ Unpaired (odd count): {leftover['name']}")
        print(f"    {leftover['url']}")
        print()

    conn.close()


def explore(no_decaf=False, exclude_urls=None):
    """For each flavor dimension, find a high/low pair that are otherwise similar."""
    conn = init_db()
    conn.row_factory = sqlite3.Row

    all_coffees = conn.execute(
        """SELECT * FROM coffees
        WHERE in_stock = 1 AND total_score IS NOT NULL AND dry_fragrance IS NOT NULL"""
    ).fetchall()
    coffees = [c for c in all_coffees if not _is_blend(c)]
    if no_decaf:
        coffees = [c for c in coffees if not _is_decaf(c)]
    tried_urls = set(r[0] for r in conn.execute("SELECT url FROM tried").fetchall())
    tried_coffees = [c for c in coffees if c["url"] in tried_urls and not _is_blend(c)]

    explore_dims = [
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
    ]

    def vec_without(row, exclude_col):
        """Vector of all dimensions EXCEPT the one we're exploring."""
        cols = [c for c in ALL_COLS if c != exclude_col]
        return [(row[c] or 0) / (10.0 if c in CUPPING_COLS else 5.0) for c in cols]

    def dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    print(f"\n{'━' * 60}")
    print("  EXPLORE — Dimension Contrast Pairs")
    print("  Find high/low pairs for each dimension, otherwise similar")
    print(f"  ({len(coffees)} in-stock coffees)")
    print(f"{'━' * 60}\n")

    for dim_name, dim_col in explore_dims:
        with_dim = [(c, c[dim_col]) for c in coffees if c[dim_col] is not None]
        if len(with_dim) < 4:
            continue

        with_dim.sort(key=lambda x: -x[1])
        high_candidates = with_dim[:5]
        low_candidates = with_dim[-5:]

        best_pair = None
        best_other_dist = float("inf")
        for hc, hv in high_candidates:
            hvec = vec_without(hc, dim_col)
            for lc, lv in low_candidates:
                if hv - lv < 1.0:
                    continue
                lvec = vec_without(lc, dim_col)
                d = dist(hvec, lvec)
                if d < best_other_dist:
                    best_other_dist = d
                    best_pair = (hc, hv, lc, lv, d)

        if not best_pair:
            continue

        hc, hv, lc, lv, other_d = best_pair

        all_vals = sorted(set(v for _, v in with_dim), reverse=True)
        top3 = [(c["name"], v) for c, v in with_dim[:3]]
        bot3 = [(c["name"], v) for c, v in with_dim[-3:]]

        h_similar = l_similar = None
        h_sim_dist = l_sim_dist = 0
        if tried_coffees:
            hvec_full = vec_without(hc, dim_col)
            lvec_full = vec_without(lc, dim_col)
            h_similar = min(
                tried_coffees, key=lambda t: dist(vec_without(t, dim_col), hvec_full)
            )
            h_sim_dist = dist(vec_without(h_similar, dim_col), hvec_full)
            l_similar = min(
                tried_coffees, key=lambda t: dist(vec_without(t, dim_col), lvec_full)
            )
            l_sim_dist = dist(vec_without(l_similar, dim_col), lvec_full)

        print(
            f"  ┌─ {dim_name} ─── range: {all_vals[-1]:.1f} to {all_vals[0]:.1f} "
            f"── pair similarity: {other_d:.2f} (lower=more isolated)"
        )
        print("  │")
        print(f"  │  TOP: {', '.join(f'{n}({v})' for n, v in top3)}")
        print(f"  │  BOT: {', '.join(f'{n}({v})' for n, v in bot3)}")
        print("  │")
        print(f"  │  ★ PAIR (most similar except for {dim_name}):")
        print(f"  │    HIGH: {hc['name']} [{dim_name}={hv}]")
        if h_similar:
            print(
                f"  │          you tried similar: {h_similar['name']} "
                f"(dist={h_sim_dist:.2f})"
            )
        print(f"  │    LOW:  {lc['name']} [{dim_name}={lv}]")
        if l_similar:
            print(
                f"  │          you tried similar: {l_similar['name']} "
                f"(dist={l_sim_dist:.2f})"
            )
        print(f"  └{'─' * 55}")
        print()

    conn.close()


def pca_reduce(vecs, n_components=None, variance_threshold=0.80):
    """PCA via SVD. Returns components, scores, explained variance, and loadings.

    Args:
        vecs: list of 22D vectors (already normalized)
        n_components: fixed number of components (overrides threshold)
        variance_threshold: keep enough components to explain this fraction

    Returns dict with:
        n_components: number of factors kept
        scores: (n_coffees × n_components) projected coordinates
        components: (n_components × 22) principal axes (loadings per dimension)
        explained_variance_ratio: fraction of variance per component
        cumulative_variance: running total of explained variance
        mean: per-dimension mean (used for centering)
    """
    import numpy as np

    X = np.array(vecs, dtype=np.float64)
    n, d = X.shape
    mean = X.mean(axis=0)
    X_centered = X - mean

    # SVD: X_centered = U @ diag(S) @ Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Explained variance
    explained_var = (S**2) / (n - 1)
    total_var = explained_var.sum()
    explained_ratio = explained_var / total_var
    cumulative = np.cumsum(explained_ratio)

    # Determine number of components
    if n_components is None:
        n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
        n_components = max(2, min(n_components, d))

    # Project data onto top components
    components = Vt[:n_components]  # (n_components × 22)
    scores = X_centered @ components.T  # (n × n_components)

    return {
        "n_components": n_components,
        "scores": scores.tolist(),
        "components": components.tolist(),
        "explained_variance_ratio": explained_ratio[:n_components].tolist(),
        "cumulative_variance": cumulative[:n_components].tolist(),
        "mean": mean.tolist(),
        "all_explained_ratio": explained_ratio.tolist(),
        "all_cumulative": cumulative.tolist(),
    }


def _interpret_component(loadings, dim_names, top_n=4):
    """Interpret a PCA component by its strongest loadings (positive and negative)."""
    pairs = list(zip(dim_names, loadings))
    pairs.sort(key=lambda x: -abs(x[1]))

    pos = [(n, v) for n, v in pairs if v > 0.15][:top_n]
    neg = [(n, v) for n, v in pairs if v < -0.15][:top_n]
    return pos, neg


def factors(
    no_decaf=False,
    exclude_urls=None,
    n_components=None,
    available_only=False,
    variance_threshold=0.80,
):
    """PCA factor extraction: find latent flavor themes that explain the variance."""
    import numpy as np

    conn = init_db()
    conn.row_factory = sqlite3.Row
    rows = get_scored_coffees(
        conn,
        no_decaf=no_decaf,
        exclude_urls=exclude_urls,
        available_only=available_only,
    )

    if len(rows) < 5:
        print("Need at least 5 scored coffees for factor analysis.")
        conn.close()
        return

    vecs = [to_vector(r) for r in rows]
    pca = pca_reduce(
        vecs, n_components=n_components, variance_threshold=variance_threshold
    )
    n_comp = pca["n_components"]
    scores = np.array(pca["scores"])
    components = np.array(pca["components"])

    # Get tried coffees for exploration gap analysis
    tried_urls = set(r[0] for r in conn.execute("SELECT url FROM tried").fetchall())

    print(f"\n{'━' * 60}")
    print(f"  PCA FACTOR ANALYSIS — {len(rows)} coffees, {n_comp} factors")
    print(f"{'━' * 60}")

    # Variance explained summary
    print("\n  ── Variance Explained ──\n")
    all_ratios = pca["all_explained_ratio"]
    all_cumul = pca["all_cumulative"]
    for i in range(min(10, len(all_ratios))):
        bar_len = int(all_ratios[i] * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " ◄ kept" if i < n_comp else ""
        print(
            f"    Factor {i + 1:2d}: {all_ratios[i]:5.1%} "
            f"(cumul {all_cumul[i]:5.1%}) {bar}{marker}"
        )
    if len(all_ratios) > 10:
        print(f"    ... {len(all_ratios) - 10} more (total dims: {len(all_ratios)})")
    print(
        f"\n    {n_comp} factors explain {pca['cumulative_variance'][-1]:.1%} of variance"
    )

    # Factor interpretation
    print("\n  ── Factor Interpretation ──")
    for i in range(n_comp):
        loadings = components[i]
        pos, neg = _interpret_component(loadings, DIM_NAMES)
        var_pct = pca["explained_variance_ratio"][i]

        print(f"\n    Factor {i + 1} ({var_pct:.1%} variance)")

        # Name the axis by its strongest contrasts
        pos_str = " + ".join(f"{n}({v:+.2f})" for n, v in pos[:3])
        neg_str = " + ".join(f"{n}({v:+.2f})" for n, v in neg[:3])
        if pos_str and neg_str:
            print(f"      (+) {pos_str}")
            print(f"      (−) {neg_str}")
            # Suggest a name
            pos_short = "/".join(n for n, _ in pos[:2])
            neg_short = "/".join(n for n, _ in neg[:2])
            print(f"      Axis: {pos_short} ←→ {neg_short}")
        elif pos_str:
            print(f"      (+) {pos_str}")
        elif neg_str:
            print(f"      (−) {neg_str}")

        # Extremes on this factor
        factor_scores = scores[:, i]
        high_idx = np.argsort(factor_scores)[-3:][::-1]
        low_idx = np.argsort(factor_scores)[:3]

        print("      High end:")
        for idx in high_idx:
            r = rows[idx]
            tried = "●" if r["url"] in tried_urls else "○"
            print(f"        {tried} {r['name']} ({factor_scores[idx]:+.2f})")
        print("      Low end:")
        for idx in low_idx:
            r = rows[idx]
            tried = "●" if r["url"] in tried_urls else "○"
            print(f"        {tried} {r['name']} ({factor_scores[idx]:+.2f})")

    # Exploration gap: where are tried vs untried in factor space?
    tried_indices = [i for i, r in enumerate(rows) if r["url"] in tried_urls]
    untried_indices = [i for i, r in enumerate(rows) if r["url"] not in tried_urls]

    if tried_indices and untried_indices:
        print("\n  ── Exploration Gaps ──")
        print("    (● = tried, ○ = untried)\n")

        tried_scores = scores[tried_indices]

        for i in range(n_comp):
            tried_range = (tried_scores[:, i].min(), tried_scores[:, i].max())

            # Find untried coffees outside the tried range on this factor
            beyond_high = [
                idx for idx in untried_indices if scores[idx, i] > tried_range[1]
            ]
            beyond_low = [
                idx for idx in untried_indices if scores[idx, i] < tried_range[0]
            ]

            if beyond_high or beyond_low:
                pos, neg = _interpret_component(components[i], DIM_NAMES)
                pos_short = "/".join(n for n, _ in pos[:2])
                neg_short = "/".join(n for n, _ in neg[:2])
                print(f"    Factor {i + 1} ({pos_short} ←→ {neg_short}):")
                print(
                    f"      Your tried range: [{tried_range[0]:+.2f}, {tried_range[1]:+.2f}]"
                )

                if beyond_high:
                    best = max(beyond_high, key=lambda idx: scores[idx, i])
                    r = rows[best]
                    print(
                        f"      Unexplored HIGH (+{pos_short}): "
                        f"{r['name']} ({scores[best, i]:+.2f})"
                    )
                    print(f"        {r['url']}")
                if beyond_low:
                    best = min(beyond_low, key=lambda idx: scores[idx, i])
                    r = rows[best]
                    print(
                        f"      Unexplored LOW (+{neg_short}): "
                        f"{r['name']} ({scores[best, i]:+.2f})"
                    )
                    print(f"        {r['url']}")
                print()

    # Dimension redundancy: which raw dimensions are essentially the same factor?
    print("  ── Dimension Redundancy ──")
    print("    Dimensions that move together (could be collapsed):\n")
    for i in range(n_comp):
        loadings = components[i]
        strong = [
            (DIM_NAMES[j], loadings[j]) for j in range(22) if abs(loadings[j]) > 0.35
        ]
        if len(strong) >= 2:
            strong.sort(key=lambda x: -x[1])
            pos_group = [n for n, v in strong if v > 0]
            neg_group = [n for n, v in strong if v < 0]
            if len(pos_group) >= 2:
                print(f"    Factor {i + 1} (+): {', '.join(pos_group)} move together")
            if len(neg_group) >= 2:
                print(f"    Factor {i + 1} (−): {', '.join(neg_group)} move together")

    print()
    conn.close()


# --- Archetypal Analysis ---


def _project_simplex(v):
    """Project vector v onto the probability simplex (non-negative, sums to 1).

    Uses the algorithm from Duchi et al. 2008.
    """
    import numpy as np

    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0)


def _project_simplex_rows(V):
    """Project each row of V onto the probability simplex (batched).

    V is (n × k). Returns (n × k) with each row on the simplex.
    Uses the Duchi et al. 2008 algorithm, vectorized across rows.
    """
    import numpy as np

    n, k = V.shape
    U = np.sort(V, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1) - 1
    indices = np.arange(1, k + 1).reshape(1, k)
    mask = U * indices > cssv
    # rho[i] = last index where condition holds for row i
    rho = k - 1 - np.argmax(mask[:, ::-1], axis=1)
    theta = cssv[np.arange(n), rho] / (rho + 1.0)
    return np.maximum(V - theta[:, np.newaxis], 0)


def archetypal_analysis(scores, n_archetypes, max_iter=200, tol=1e-6, seed=42):
    """Find archetypes in PCA-reduced space via alternating constrained least squares.

    Each data point x_i ≈ sum_k alpha_ik * z_k  (alpha: mixture weights, simplex)
    Each archetype z_k = sum_i beta_ki * x_i    (beta: data weights, simplex)

    Uses Lipschitz-based projected gradient for each simplex subproblem.

    Args:
        scores: (n × d) numpy array of PCA-projected data
        n_archetypes: number of archetypes to find
        max_iter: maximum iterations
        tol: convergence tolerance (relative RSS change)
        seed: random seed

    Returns dict with:
        archetypes: (n_archetypes × d) archetype coordinates in PCA space
        alpha: (n × n_archetypes) mixture weights per data point
        beta: (n_archetypes × n) data weights per archetype
        rss: final residual sum of squares
        rss_history: RSS per iteration
    """
    import numpy as np

    X = np.array(scores, dtype=np.float64)
    n, d = X.shape
    k = n_archetypes

    def _solve_simplex_qp(A, b, w_init=None, max_inner=80):
        """Solve min ||A @ w - b||^2 s.t. w >= 0, sum(w) = 1.

        Uses projected gradient with Lipschitz step size.
        A is (d × m), b is (d,), w is (m,).
        """
        m = A.shape[1]
        if w_init is not None:
            w = w_init.copy()
        else:
            w = np.ones(m) / m

        AtA = A.T @ A
        Atb = A.T @ b
        L = np.linalg.norm(AtA, ord=2) + 1e-8
        lr = 1.0 / L

        for _ in range(max_inner):
            grad = AtA @ w - Atb
            w_new = _project_simplex(w - lr * grad)
            if np.linalg.norm(w_new - w) < 1e-8:
                break
            w = w_new
        return w

    def _solve_alpha_batched(Z, X, alpha_init, max_inner=80):
        """Vectorized alpha update: solve all n simplex QPs simultaneously.

        min ||X - alpha @ Z||^2 s.t. each row of alpha on simplex.
        Equivalent to per-row: min ||Z.T @ w - x_i||^2 s.t. w on simplex.

        Z is (k × d), X is (n × d), alpha_init is (n × k).
        """
        ZZT = Z @ Z.T  # (k × k) — shared Hessian for all points
        XZT = X @ Z.T  # (n × k) — all linear terms at once
        L = np.linalg.norm(ZZT, ord=2) + 1e-8
        lr = 1.0 / L

        alpha = alpha_init.copy()
        for _ in range(max_inner):
            # grad[i] = ZZT @ alpha[i] - XZT[i] for all i simultaneously
            grad = alpha @ ZZT - XZT  # (n × k)
            alpha_new = _project_simplex_rows(alpha - lr * grad)
            # Convergence check: max row movement
            if np.max(np.linalg.norm(alpha_new - alpha, axis=1)) < 1e-8:
                break
            alpha = alpha_new
        return alpha

    # Initialize with furthest-first traversal
    centroid = X.mean(axis=0)
    dists = np.linalg.norm(X - centroid, axis=1)
    indices = [int(np.argmax(dists))]
    for _ in range(k - 1):
        min_dists = np.min(
            [np.linalg.norm(X - X[idx], axis=1) for idx in indices], axis=0
        )
        next_idx = int(np.argmax(min_dists))
        indices.append(next_idx)

    # Initialize archetypes as those extreme data points
    Z = X[indices].copy()

    # Initialize beta: each archetype is one data point
    beta = np.zeros((k, n))
    for ki, idx in enumerate(indices):
        beta[ki, idx] = 1.0

    # Initialize alpha (batched)
    alpha_init = np.full((n, k), 1.0 / k)
    alpha = _solve_alpha_batched(Z, X, alpha_init)

    rss_history = []

    for iteration in range(max_iter):
        # --- Step 1: Update alpha given Z (batched) ---
        alpha = _solve_alpha_batched(Z, X, alpha)

        # --- Step 2: Update Z (archetypes) given alpha ---
        # Optimal unconstrained Z: solve alpha^T @ alpha @ Z = alpha^T @ X
        AtA = alpha.T @ alpha + np.eye(k) * 1e-6
        Z_target = np.linalg.solve(AtA, alpha.T @ X)

        # --- Step 3: Update beta so that Z = beta @ X approximates Z_target ---
        for j in range(k):
            # Only consider candidate points near the target
            dists_to_target = np.linalg.norm(X - Z_target[j], axis=1)
            top_n_candidates = min(40, n)
            top_candidates = np.argsort(dists_to_target)[:top_n_candidates]
            current_nonzero = np.nonzero(beta[j] > 0.005)[0]
            candidate_idx = np.unique(np.concatenate([top_candidates, current_nonzero]))

            X_sub = X[candidate_idx].T  # (d × m)
            beta_sub = _solve_simplex_qp(X_sub, Z_target[j])

            beta[j] = 0.0
            beta[j, candidate_idx] = beta_sub

        # Recompute archetypes from beta
        Z = beta @ X

        # Compute RSS
        residual = X - alpha @ Z
        rss = float(np.sum(residual**2))
        rss_history.append(rss)

        if len(rss_history) > 1:
            rel_change = abs(rss_history[-2] - rss) / (rss_history[-2] + 1e-10)
            if rel_change < tol:
                break

    return {
        "archetypes": Z.tolist(),
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "rss": rss,
        "rss_history": rss_history,
        "n_iter": len(rss_history),
    }


def _pick_n_archetypes(scores, candidates=None, seed=42):
    """Auto-select number of archetypes by RSS elbow (diminishing returns)."""

    if candidates is None:
        candidates = [3, 4, 5, 6, 7]

    results = {}
    for k in candidates:
        aa = archetypal_analysis(scores, k, max_iter=100, seed=seed)
        results[k] = aa["rss"]

    # Find elbow: biggest drop in marginal RSS improvement
    ks = sorted(results.keys())
    if len(ks) < 3:
        return ks[0], results

    drops = [results[ks[i]] - results[ks[i + 1]] for i in range(len(ks) - 1)]
    # Pick the k where the next marginal drop is < 50% of the current drop
    best_k = ks[-1]
    for i in range(len(drops) - 1):
        if drops[i + 1] < drops[i] * 0.5:
            best_k = ks[i + 1]
            break

    return best_k, results


def archetypes(
    no_decaf=False,
    exclude_urls=None,
    n_archetypes=None,
    available_only=False,
    variance_threshold=0.80,
):
    """Archetypal analysis: find extreme flavor styles and express coffees as mixtures."""
    import numpy as np

    conn = init_db()
    conn.row_factory = sqlite3.Row
    rows = get_scored_coffees(
        conn,
        no_decaf=no_decaf,
        exclude_urls=exclude_urls,
        available_only=available_only,
    )

    if len(rows) < 10:
        print("Need at least 10 scored coffees for archetypal analysis.")
        conn.close()
        return

    # PCA reduction first
    vecs = [to_vector(r) for r in rows]
    pca = pca_reduce(vecs, variance_threshold=variance_threshold)
    n_pca = pca["n_components"]
    scores = np.array(pca["scores"])
    components = np.array(pca["components"])

    # Auto-select or use specified number of archetypes
    if n_archetypes is None:
        print(
            f"\n  Auto-selecting archetype count (testing 3–7 in {n_pca}D PCA space)..."
        )
        n_archetypes, rss_by_k = _pick_n_archetypes(scores)
        print(
            f"  RSS by k: {', '.join(f'{k}={v:.1f}' for k, v in sorted(rss_by_k.items()))}"
        )
        print(f"  Selected k={n_archetypes}")

    # Run full archetypal analysis
    aa = archetypal_analysis(scores, n_archetypes, max_iter=300)
    arch_scores = np.array(aa["archetypes"])  # (k × n_pca) in PCA space
    alpha = np.array(aa["alpha"])  # (n × k) mixture weights

    # Project archetypes back to 22D for interpretation
    arch_22d = arch_scores @ components + np.array(pca["mean"])

    # Get tried info
    tried_urls = set(r[0] for r in conn.execute("SELECT url FROM tried").fetchall())
    tried_rows_full = conn.execute(
        """SELECT c.*, t.rating FROM tried t
        JOIN coffees c ON t.url = c.url WHERE c.dry_fragrance IS NOT NULL"""
    ).fetchall()
    tried_ratings = {r["url"]: r["rating"] for r in tried_rows_full}

    print(f"\n{'━' * 60}")
    print(f"  ARCHETYPAL ANALYSIS — {len(rows)} coffees, {n_archetypes} archetypes")
    print(f"  ({n_pca}D PCA space, {aa['n_iter']} iterations, RSS={aa['rss']:.1f})")
    print(f"{'━' * 60}")

    # --- Name and describe each archetype ---
    print("\n  ── Archetypes (Pure Flavor Styles) ──")

    # Compute centroid in 22D for contrast
    mean_22d = np.array(pca["mean"])
    stddevs = compute_dim_stddevs(vecs)
    ds = _dscale()

    # Available pool: in-stock, untried, non-blend (for recommendations/contrast)
    avail_indices = [
        i for i, r in enumerate(rows) if r["in_stock"] and r["url"] not in tried_urls
    ]
    avail_pool = [(i, rows[i]) for i in avail_indices]

    def _find_contrast(ref_idx, pool_indices):
        """Find contrast: nearest in-stock coffee to the antipode in PCA space.

        Reflects the reference through the origin (PCA space is centered),
        then finds the closest coffee to that mirror point. This yields
        a true inverse profile — high where you're low, low where you're high.
        """
        ref_score = scores[ref_idx]
        antipode = -ref_score  # PCA space is zero-centered
        best_idx = None
        best_dist = float("inf")
        for idx in pool_indices:
            if idx == ref_idx:
                continue
            d = float(np.linalg.norm(scores[idx] - antipode))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def _print_contrast_pair(ref_idx, pool_indices, indent="      "):
        """Print the contrast coffee: nearest to antipode in PCA space."""
        contrast_idx = _find_contrast(ref_idx, pool_indices)
        if contrast_idx is None:
            return
        contrast_coffee = rows[contrast_idx]
        contrast_mix = alpha[contrast_idx]
        parts = sorted(
            [(archetype_names[a], contrast_mix[a]) for a in range(n_archetypes)],
            key=lambda x: -x[1],
        )
        parts_str = " + ".join(f"{n} {w:.0%}" for n, w in parts if w > 0.05)
        # Show the key 22D differences
        ref_vec = vecs[ref_idx]
        con_vec = vecs[contrast_idx]
        diffs_22d = [
            (DIM_NAMES[j], con_vec[j] * ds[j] - ref_vec[j] * ds[j])
            for j in range(22)
            if abs(con_vec[j] - ref_vec[j]) > 0.5 * stddevs[j]
        ]
        diffs_22d.sort(key=lambda x: -abs(x[1]))
        more = [f"{n}+{d:.1f}" for n, d in diffs_22d if d > 0][:3]
        less = [f"{n}{d:.1f}" for n, d in diffs_22d if d < 0][:2]
        print(f"{indent}Contrast: {contrast_coffee['name']}")
        print(f"{indent}  {contrast_coffee['url']}")
        print(f"{indent}  Mix: {parts_str}")
        desc = []
        if more:
            desc.append(f"more {', '.join(more)}")
        if less:
            desc.append(f"less {', '.join(less)}")
        if desc:
            print(f"{indent}  vs recommendation: {'; '.join(desc)}")

    # First pass: compute archetype names (needed by contrast pair display)
    archetype_names = []
    for ai in range(n_archetypes):
        arch_vec = arch_22d[ai]
        diffs = [(DIM_NAMES[j], (arch_vec[j] - mean_22d[j]) * ds[j]) for j in range(22)]
        diffs.sort(key=lambda x: -abs(x[1]))
        top_pos = [(n, d) for n, d in diffs if d > 0.3 * stddevs[0]][:3]
        name_parts = [n for n, _ in top_pos[:2]]
        if not name_parts:
            name_parts = ["Balanced"]
        archetype_names.append("/".join(name_parts))

    # Second pass: display each archetype with recommendations
    for ai in range(n_archetypes):
        arch_vec = arch_22d[ai]
        # Find dimensions where this archetype deviates most from mean
        diffs = [(DIM_NAMES[j], (arch_vec[j] - mean_22d[j]) * ds[j]) for j in range(22)]
        diffs.sort(key=lambda x: -abs(x[1]))

        # Name by top 2-3 distinguishing traits
        top_pos = [(n, d) for n, d in diffs if d > 0.3 * stddevs[0]][:3]
        top_neg = [(n, d) for n, d in diffs if d < -0.3 * stddevs[0]][:2]

        archetype_name = archetype_names[ai]

        # Find purest expression from IN-STOCK coffees first, fall back to all
        arch_dominance = alpha[:, ai]
        avail_for_arch = [(idx, arch_dominance[idx]) for idx in avail_indices]
        avail_for_arch.sort(key=lambda x: -x[1])

        if avail_for_arch and avail_for_arch[0][1] > 0.3:
            purest_idx = avail_for_arch[0][0]
        else:
            # Fall back to absolute purest (may be out of stock)
            purest_idx = int(np.argmax(arch_dominance))
        purest_coffee = rows[purest_idx]
        purest_weight = arch_dominance[purest_idx]

        print(f"\n    Archetype {ai + 1}: {archetype_name}")
        high_str = ", ".join(f"{n}+{d:.2f}" for n, d in top_pos[:4])
        low_str = ", ".join(f"{n}{d:.2f}" for n, d in top_neg[:3])
        if high_str:
            print(f"      High: {high_str}")
        if low_str:
            print(f"      Low:  {low_str}")
        print(
            f"      Purest expression: {purest_coffee['name']} "
            f"({purest_weight:.0%} this archetype)"
        )
        print(f"        {purest_coffee['url']}")
        stock_mark = "✅" if purest_coffee["in_stock"] else "❌"
        tried_mark = "●" if purest_coffee["url"] in tried_urls else "○"
        print(f"        {tried_mark} tried  {stock_mark} stock")

        # Contrast pair for the purest expression
        if purest_coffee["in_stock"] and avail_indices:
            _print_contrast_pair(purest_idx, avail_indices, indent="        ")

        # Show top 3 in-stock coffees most dominated by this archetype
        top_avail = avail_for_arch[:4]
        others = [(rows[idx], w) for idx, w in top_avail if idx != purest_idx][:3]
        if others:
            print("      Also strong (in stock):")
            for coffee, weight in others:
                print(f"        ○ {coffee['name']} ({weight:.0%})")

    # --- Express tried coffees as archetype mixtures ---
    print("\n  ── Your Tried Coffees as Archetype Mixtures ──\n")

    tried_indices = [(i, r) for i, r in enumerate(rows) if r["url"] in tried_urls]
    if tried_indices:
        for idx, coffee in tried_indices:
            mix = alpha[idx]
            rating = tried_ratings.get(coffee["url"], "?")
            sym = {"+": "👍", "0": "😐", "-": "👎"}.get(rating, "?")
            # Show mixture as bar
            parts = sorted(
                [(archetype_names[ai], mix[ai]) for ai in range(n_archetypes)],
                key=lambda x: -x[1],
            )
            parts_str = " + ".join(f"{name} {w:.0%}" for name, w in parts if w > 0.05)
            print(f"    {sym} {coffee['name']}")
            print(f"      {parts_str}")

        # Compute tried archetype coverage
        print("\n  ── Archetype Coverage ──\n")
        tried_alpha = np.array([alpha[idx] for idx, _ in tried_indices])
        # Average exposure to each archetype across tried coffees
        mean_exposure = tried_alpha.mean(axis=0)
        max_exposure = tried_alpha.max(axis=0)

        exposure_ranked = sorted(
            [
                (archetype_names[ai], mean_exposure[ai], max_exposure[ai])
                for ai in range(n_archetypes)
            ],
            key=lambda x: -x[1],
        )
        for name, avg, mx in exposure_ranked:
            bar_len = int(avg * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            status = ""
            if mx < 0.3:
                status = " ⚡ UNEXPLORED"
            elif avg < 0.15:
                status = " △ lightly explored"
            print(f"    {name:20s} avg={avg:.0%} max={mx:.0%} {bar}{status}")
    else:
        print("    (no tried coffees with flavor data)")

    # --- Recommendations: highest-weight untried for underexplored archetypes ---
    print("\n  ── Recommendations (explore underexplored archetypes) ──\n")

    tried_alpha_avg = None
    if tried_indices:
        tried_alpha_arr = np.array([alpha[idx] for idx, _ in tried_indices])
        tried_alpha_avg = tried_alpha_arr.mean(axis=0)

    if tried_alpha_avg is not None and avail_pool:
        # Rank archetypes by how underexplored they are
        arch_order = np.argsort(tried_alpha_avg)  # least explored first

        for ai in arch_order[:3]:  # top 3 most underexplored
            name = archetype_names[ai]
            # Find best in-stock untried coffee with highest weight in this archetype
            candidates = [(idx, r, alpha[idx, ai]) for idx, r in avail_pool]
            candidates.sort(key=lambda x: -x[2])
            if candidates:
                best_idx, best_coffee, best_w = candidates[0]
                mix = alpha[best_idx]
                parts = sorted(
                    [(archetype_names[a], mix[a]) for a in range(n_archetypes)],
                    key=lambda x: -x[1],
                )
                parts_str = " + ".join(f"{n} {w:.0%}" for n, w in parts if w > 0.05)
                print(
                    f"    To explore [{name}] (your avg exposure: {tried_alpha_avg[ai]:.0%}):"
                )
                print(f"      {best_coffee['name']} ({best_w:.0%} this archetype)")
                print(f"      {best_coffee['url']}")
                print(f"      Full mix: {parts_str}")
                # Contrast pair: most different available coffee from this recommendation
                _print_contrast_pair(best_idx, avail_indices, indent="      ")
                print()

    # --- Most complex coffees (high entropy, in-stock) ---
    print("  ── Most Complex (balanced across archetypes, in stock) ──\n")
    entropies = []
    for idx in avail_indices:
        mix = alpha[idx]
        # Shannon entropy (higher = more evenly spread)
        ent = -sum(w * np.log(w + 1e-10) for w in mix if w > 0.01)
        entropies.append((idx, ent))
    entropies.sort(key=lambda x: -x[1])

    max_entropy = np.log(n_archetypes)  # theoretical max
    for idx, ent in entropies[:5]:
        coffee = rows[idx]
        mix = alpha[idx]
        parts = sorted(
            [(archetype_names[ai], mix[ai]) for ai in range(n_archetypes)],
            key=lambda x: -x[1],
        )
        parts_str = " + ".join(f"{n} {w:.0%}" for n, w in parts if w > 0.05)
        print(f"    {coffee['name']} (entropy {ent:.2f}/{max_entropy:.2f})")
        print(f"       {coffee['url']}")
        print(f"       {parts_str}")

    print()
    conn.close()


# --- UMAP 2D Flavor Map ---


def _write_map_html(html_path, png_filename, hotspots):
    """Write an HTML file that overlays clickable hotspots on the flavor map PNG."""
    import html as html_mod

    spots_html = []
    for spot in hotspots:
        name_escaped = html_mod.escape(spot["name"], quote=True)
        arch_escaped = html_mod.escape(spot["archetype"], quote=True)
        spots_html.append(
            f'    <a href="{spot["url"]}" target="_blank" class="spot" '
            f'style="left:{spot["x"]:.3f}%;top:{spot["y"]:.3f}%" '
            f'data-name="{name_escaped}" data-arch="{arch_escaped}"></a>'
        )
    spots_str = "\n".join(spots_html)

    content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Coffee Flavor Map</title>
<style>
  body {{
    margin: 0;
    background: #0f0f1a;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  }}
  .map-container {{
    position: relative;
    display: inline-block;
    max-width: 100%;
    width: 1400px;
  }}
  .map-container img {{
    display: block;
    width: 100%;
    height: auto;
  }}
  .spot {{
    position: absolute;
    width: 20px;
    height: 20px;
    margin-left: -10px;
    margin-top: -10px;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.15s, transform 0.15s;
  }}
  .spot:hover {{
    opacity: 1;
    background: rgba(255, 255, 255, 0.3);
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.6);
    transform: scale(1.3);
  }}
  .tooltip {{
    position: fixed;
    background: #2a2a4a;
    color: #fff;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    z-index: 1000;
    max-width: 300px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    white-space: nowrap;
  }}
  .tooltip .name {{
    font-weight: 600;
    margin-bottom: 3px;
  }}
  .tooltip .arch {{
    color: #aaa;
    font-size: 11px;
  }}
</style>
</head>
<body>
<div class="map-container">
  <img src="{png_filename}" alt="Coffee Flavor Map">
{spots_str}
</div>
<div class="tooltip" id="tooltip"></div>
<script>
  const tooltip = document.getElementById('tooltip');
  document.querySelectorAll('.spot').forEach(spot => {{
    spot.addEventListener('mouseenter', e => {{
      tooltip.innerHTML = '<div class="name">' + spot.dataset.name + '</div>'
                        + '<div class="arch">' + spot.dataset.arch + '</div>';
      tooltip.style.opacity = '1';
    }});
    spot.addEventListener('mousemove', e => {{
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top = (e.clientY + 14) + 'px';
    }});
    spot.addEventListener('mouseleave', () => {{
      tooltip.style.opacity = '0';
    }});
  }});
</script>
</body>
</html>
"""
    with open(html_path, "w") as f:
        f.write(content)


def flavor_map_slider(
    no_decaf=False,
    exclude_urls=None,
    available_only=False,
    variance_threshold=0.80,
    output_path="flavor-map-slider.html",
    n_neighbors=10,
    min_dist=0.3,
    n_steps=11,
):
    """Generate an interactive HTML flavor map with a text-weight slider.

    Pre-computes UMAP at n_steps evenly spaced text weights (0.0 to 1.0),
    Procrustes-aligns all frames to the midpoint, then emits a self-contained
    HTML file with SVG rendering and a slider that interpolates positions.

    Archetypal analysis runs per frame on a blended feature space (numeric PCA
    + text PCA scaled by the step's weight), so archetype count and assignment
    evolve as text influence increases.
    """
    import json
    import re

    import numpy as np
    import umap

    conn = init_db()
    conn.row_factory = sqlite3.Row
    rows = get_scored_coffees(
        conn,
        no_decaf=no_decaf,
        exclude_urls=exclude_urls,
        available_only=available_only,
    )

    if len(rows) < 15:
        print("Need at least 15 scored coffees for UMAP projection.")
        conn.close()
        return

    # PCA reduction
    vecs = [to_vector(r) for r in rows]
    pca = pca_reduce(vecs, variance_threshold=variance_threshold)
    scores = np.array(pca["scores"])
    n_pca = pca["n_components"]
    components = np.array(pca["components"])
    mean_22d = np.array(pca["mean"])

    # Load text embeddings (required for slider)
    embeddings_dict = load_embeddings(conn)
    if not embeddings_dict:
        print("Slider map requires text embeddings. Run: python embed.py build")
        conn.close()
        return

    # Build text PCA component (same approach as concat mode in flavor_map)
    n = len(rows)
    text_vecs = []
    for row in rows:
        emb = embeddings_dict.get(row["url"])
        if emb is not None:
            text_vecs.append(emb)
        else:
            text_vecs.append(np.zeros(384, dtype=np.float32))
    text_matrix = np.array(text_vecs, dtype=np.float64)

    text_mean = text_matrix.mean(axis=0)
    text_centered = text_matrix - text_mean
    U_t, S_t, Vt_t = np.linalg.svd(text_centered, full_matrices=False)
    text_var = (S_t**2) / (n - 1)
    text_cumvar = np.cumsum(text_var / text_var.sum())
    n_text_components = int(np.searchsorted(text_cumvar, 0.80) + 1)
    n_text_components = max(3, min(n_text_components, 10))
    text_scores = text_centered @ Vt_t[:n_text_components].T

    # Normalize text to match numeric variance scale
    num_scale = np.std(scores)
    text_scale = np.std(text_scores)
    if text_scale > 0:
        text_scores_scaled = text_scores * (num_scale / text_scale)
    else:
        text_scores_scaled = text_scores

    # Pre-compute UMAP at each text weight step
    weights = np.linspace(0.0, 1.0, n_steps)
    frames = []

    for step_i, tw in enumerate(weights):
        logger.info(
            "computing UMAP frame %d/%d (text_weight=%.2f)", step_i + 1, n_steps, tw
        )
        # Build blended distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                nd = float(np.linalg.norm(scores[i] - scores[j]))
                emb_i = embeddings_dict.get(rows[i]["url"])
                emb_j = embeddings_dict.get(rows[j]["url"])
                if emb_i is not None and emb_j is not None:
                    td = 1.0 - float(emb_i @ emb_j)
                else:
                    td = nd
                blended = (1.0 - tw) * nd + tw * td
                dist_matrix[i, j] = blended
                dist_matrix[j, i] = blended

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="precomputed",
            random_state=42,
        )
        embedding = reducer.fit_transform(dist_matrix)
        frames.append(embedding)

    # Procrustes-align all frames to the midpoint frame
    ref_idx = n_steps // 2
    ref = frames[ref_idx].copy()

    ref_centered = ref - ref.mean(axis=0)
    ref_scale = np.sqrt((ref_centered**2).sum())
    ref_norm = ref_centered / ref_scale

    aligned_frames = []
    for fi, frame in enumerate(frames):
        if fi == ref_idx:
            aligned_frames.append(ref_norm)
            continue
        target = frame - frame.mean(axis=0)
        target_scale = np.sqrt((target**2).sum())
        target_norm = target / target_scale

        M = ref_norm.T @ target_norm
        U, S, Vt = np.linalg.svd(M)
        d = np.linalg.det(U @ Vt)
        D = np.diag([1.0, 1.0 if d > 0 else -1.0])
        R = U @ D @ Vt
        aligned = target_norm @ R.T
        aligned_frames.append(aligned)

    # --- Per-frame archetypal analysis on blended feature space ---
    ds = _dscale()
    stddevs = compute_dim_stddevs(vecs)
    avail_indices = [i for i, row in enumerate(rows) if row["in_stock"]]

    # Get cupping notes for text keywords in archetype naming
    notes_by_url = {}
    for row in conn.execute(
        "SELECT url, cupping_notes FROM coffees WHERE cupping_notes IS NOT NULL"
    ).fetchall():
        notes_by_url[row["url"]] = row["cupping_notes"]

    tab10 = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    frame_archetypes = []
    for step_i, tw in enumerate(weights):
        logger.info(
            "computing archetypes for frame %d/%d (text_weight=%.2f)",
            step_i + 1,
            n_steps,
            tw,
        )
        # Blended feature space: numeric + text scaled by weight
        if tw > 0:
            arch_input = np.hstack([scores, text_scores_scaled * tw])
        else:
            arch_input = scores

        n_arch, _ = _pick_n_archetypes(arch_input, seed=42)
        aa = archetypal_analysis(arch_input, n_arch, max_iter=300)
        alpha = np.array(aa["alpha"])
        dominant = np.argmax(alpha, axis=1)
        arch_coords = np.array(aa["archetypes"])

        # Name archetypes from their numeric PCA component
        arch_numeric = arch_coords[:, :n_pca]
        arch_22d = arch_numeric @ components + mean_22d

        names = []
        for ai in range(n_arch):
            arch_vec = arch_22d[ai]
            diffs = [
                (DIM_NAMES[j], (arch_vec[j] - mean_22d[j]) * ds[j]) for j in range(22)
            ]
            diffs.sort(key=lambda x: -abs(x[1]))
            top_pos = [nm for nm, d in diffs if d > 0.3 * stddevs[0]][:2]
            names.append("/".join(top_pos) if top_pos else "Balanced")

        # Add text keywords to names when text has influence
        if tw > 0:
            names = _add_text_keywords(names, alpha, rows, n_arch, conn)

        # Center indices (highest alpha among in-stock)
        centers = []
        for ai in range(n_arch):
            best_idx = None
            best_w = 0
            for i in avail_indices:
                if alpha[i, ai] > best_w:
                    best_w = alpha[i, ai]
                    best_idx = i
            if best_idx is None:
                best_idx = int(np.argmax(alpha[:, ai]))
            centers.append(best_idx)

        colors = [tab10[i % len(tab10)] for i in range(n_arch)]

        frame_archetypes.append(
            {
                "n_arch": n_arch,
                "archetypes": [
                    {"name": names[i], "color": colors[i]} for i in range(n_arch)
                ],
                "dominant": [int(d) for d in dominant],
                "centers": centers,
            }
        )

    # Compute per-coffee numeric profile and text keywords
    pop_means = [0.0] * 22
    pop_stds = [0.0] * 22
    for j in range(22):
        vals = [v[j] for v in vecs]
        pop_means[j] = sum(vals) / len(vals)
        var = sum((x - pop_means[j]) ** 2 for x in vals) / len(vals)
        pop_stds[j] = var**0.5 if var > 0 else 0.01

    conn.close()

    coffees_json = []
    for i, row in enumerate(rows):
        vec = vecs[i]
        z_scores = [
            (DIM_NAMES[j], (vec[j] - pop_means[j]) / pop_stds[j]) for j in range(22)
        ]
        z_scores.sort(key=lambda x: -x[1])
        numeric_profile = [nm for nm, z in z_scores[:3] if z > 0.3]
        if not numeric_profile:
            numeric_profile = [z_scores[0][0]]

        notes = notes_by_url.get(row["url"], "")
        text_keywords = []
        if notes:
            words = re.findall(r"[a-z]{4,}", notes.lower())
            words = [w for w in words if w not in _TEXT_STOPWORDS]
            freq = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            ranked = sorted(freq.items(), key=lambda x: -x[1])
            text_keywords = [w for w, _ in ranked[:5]]

        coffees_json.append(
            {
                "name": row["name"],
                "url": row["url"],
                "in_stock": bool(row["in_stock"]),
                "numeric": ", ".join(numeric_profile),
                "text": ", ".join(text_keywords) if text_keywords else "",
            }
        )

    # Build frames JSON
    frames_json = []
    for frame in aligned_frames:
        frames_json.append([[round(float(x), 5), round(float(y), 5)] for x, y in frame])

    data_json = json.dumps(
        {
            "steps": [round(float(w), 2) for w in weights],
            "frames": frames_json,
            "coffees": coffees_json,
            "frame_archetypes": frame_archetypes,
        },
        separators=(",", ":"),
    )

    title = f"Sweet Maria's Coffee Flavor Map Explorer, {date.today().strftime('%B %d, %Y')}"
    html_content = _slider_html_template(data_json, title)
    with open(output_path, "w") as f:
        f.write(html_content)

    arch_counts = [fa["n_arch"] for fa in frame_archetypes]
    print(f"\n  Saved: {output_path}")
    print(f"  {n} coffees, {n_steps} weight steps")
    print(f"  Archetypes per step: {arch_counts}")
    print(f"  UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}")
    print(f"  Procrustes-aligned to midpoint (weight={weights[ref_idx]:.1f})")
    print()


def _slider_html_template(data_json, title):
    """Return the self-contained HTML string for the interactive slider map."""
    return (
        """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>"""
        + title
        + """</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: #0f0f1a;
  color: #eee;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-height: 100vh;
  padding: 20px;
}
h1 {
  font-size: 18px;
  font-weight: 500;
  margin-bottom: 12px;
  color: #ccc;
}
.controls {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
  background: #1a1a2e;
  padding: 12px 24px;
  border-radius: 8px;
  border: 1px solid #333;
}
.controls label {
  font-size: 13px;
  color: #aaa;
}
.controls input[type=range] {
  width: 300px;
  accent-color: #5b8def;
}
.controls .weight-val {
  font-size: 15px;
  font-weight: 600;
  color: #5b8def;
  min-width: 36px;
  text-align: center;
}
.controls .endpoints {
  font-size: 11px;
  color: #888;
}
.info-bar {
  font-size: 12px;
  color: #aaa;
  margin-bottom: 8px;
}
.info-bar span { color: #5b8def; font-weight: 600; }
.map-wrap {
  position: relative;
  width: 900px;
  height: 700px;
  background: #1a1a2e;
  border-radius: 8px;
  border: 1px solid #333;
  overflow: hidden;
}
svg {
  width: 100%;
  height: 100%;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 12px;
  padding: 10px 16px;
  background: #1a1a2e;
  border-radius: 8px;
  border: 1px solid #333;
  max-width: 900px;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: #ccc;
}
.legend-swatch {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex-shrink: 0;
}
.tooltip {
  position: fixed;
  background: #2a2a4a;
  color: #fff;
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 12px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.12s;
  z-index: 1000;
  max-width: 320px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.6);
  line-height: 1.4;
}
.tooltip .tt-name { font-weight: 600; margin-bottom: 4px; font-size: 13px; }
.tooltip .tt-desc { margin-bottom: 3px; }
.tooltip .tt-desc .label { color: #aaa; font-size: 10px; text-transform: uppercase; }
.tooltip .tt-desc .value { color: #ddd; }
.tooltip .tt-arch { color: #aaa; font-size: 11px; margin-bottom: 2px; }
.tooltip .tt-status { color: #8f8; font-size: 11px; margin-top: 4px; }
</style>
</head>
<body>
<h1>"""
        + title
        + """</h1>
<div class="controls">
  <span class="endpoints">Cupping scores</span>
  <input type="range" id="slider" min="0" max="1" step="0.01" value="0.00">
  <span class="weight-val" id="weight-display">0.00</span>
  <span class="endpoints">Cupping notes</span>
</div>
<div class="info-bar" id="info-bar">Archetypes: <span id="arch-count">\u2014</span></div>
<div class="map-wrap">
  <svg id="map" xmlns="http://www.w3.org/2000/svg"></svg>
</div>
<div class="legend" id="legend"></div>
<div class="tooltip" id="tooltip"></div>

<script>
const DATA = """
        + data_json
        + """;

const svg = document.getElementById('map');
const slider = document.getElementById('slider');
const weightDisplay = document.getElementById('weight-display');
const tooltip = document.getElementById('tooltip');
const legendEl = document.getElementById('legend');
const archCountEl = document.getElementById('arch-count');

const { steps, frames, coffees, frame_archetypes } = DATA;
const N = coffees.length;
const nSteps = steps.length;
const maxArch = Math.max(...frame_archetypes.map(fa => fa.n_arch));
let currentWeight = 0.0;
let currentFrameIdx = -1;

// Track current interpolated positions
const positions = new Array(N);
for (let i = 0; i < N; i++) positions[i] = [0, 0];

// Compute global bounds across all frames for stable viewport
let gMinX = Infinity, gMaxX = -Infinity, gMinY = Infinity, gMaxY = -Infinity;
for (const frame of frames) {
  for (const [x, y] of frame) {
    if (x < gMinX) gMinX = x;
    if (x > gMaxX) gMaxX = x;
    if (y < gMinY) gMinY = y;
    if (y > gMaxY) gMaxY = y;
  }
}
const pad = 0.08;
const rangeX = gMaxX - gMinX || 1;
const rangeY = gMaxY - gMinY || 1;
gMinX -= rangeX * pad;
gMaxX += rangeX * pad;
gMinY -= rangeY * pad;
gMaxY += rangeY * pad;

const W = 900, H = 700;

function toSVG(x, y) {
  const sx = ((x - gMinX) / (gMaxX - gMinX)) * W;
  const sy = H - ((y - gMinY) / (gMaxY - gMinY)) * H;
  return [sx, sy];
}

// --- SVG layer groups for z-ordering ---
const lineGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
const circleGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
const markerGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
svg.appendChild(lineGroup);
svg.appendChild(circleGroup);
svg.appendChild(markerGroup);

// Dashed lines for center-contrast pairs (up to maxArch)
const contrastLines = [];
for (let ai = 0; ai < maxArch; ai++) {
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('stroke-width', '1.5');
  line.setAttribute('stroke-dasharray', '6,4');
  line.setAttribute('opacity', '0');
  lineGroup.appendChild(line);
  contrastLines.push(line);
}

// Coffee circles
const circles = [];
for (let i = 0; i < N; i++) {
  const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  const coffee = coffees[i];

  let r, baseOpacity, strokeW, stroke;
  if (coffee.in_stock) {
    r = 4.5; baseOpacity = 0.85; strokeW = 0; stroke = 'none';
  } else {
    r = 3; baseOpacity = 0.3; strokeW = 0; stroke = 'none';
  }

  c.setAttribute('r', r);
  c.setAttribute('opacity', baseOpacity);
  c.setAttribute('stroke', stroke);
  c.setAttribute('stroke-width', strokeW);
  c.style.cursor = coffee.in_stock ? 'pointer' : 'default';
  c.dataset.idx = i;
  c.dataset.baseOpacity = baseOpacity;
  circleGroup.appendChild(c);
  circles.push(c);
}

// Center star markers (up to maxArch, hide unused)
const centerStars = [];
for (let ai = 0; ai < maxArch; ai++) {
  const star = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  star.textContent = '\\u2605';
  star.setAttribute('font-size', '18');
  star.setAttribute('stroke', '#fff');
  star.setAttribute('stroke-width', '0.5');
  star.setAttribute('text-anchor', 'middle');
  star.setAttribute('dominant-baseline', 'central');
  star.setAttribute('pointer-events', 'none');
  star.setAttribute('opacity', '0');
  markerGroup.appendChild(star);
  centerStars.push(star);
}

// Contrast diamond markers (up to maxArch)
const contrastDiamonds = [];
for (let ai = 0; ai < maxArch; ai++) {
  const diamond = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  diamond.textContent = '\\u25c6';
  diamond.setAttribute('font-size', '14');
  diamond.setAttribute('stroke', '#fff');
  diamond.setAttribute('stroke-width', '0.4');
  diamond.setAttribute('text-anchor', 'middle');
  diamond.setAttribute('dominant-baseline', 'central');
  diamond.setAttribute('pointer-events', 'none');
  diamond.setAttribute('opacity', '0');
  markerGroup.appendChild(diamond);
  contrastDiamonds.push(diamond);
}

// Find contrast for each archetype: farthest in-stock point from center
function findContrasts(fa) {
  const contrasts = [];
  for (let ai = 0; ai < fa.n_arch; ai++) {
    const ci = fa.centers[ai];
    const cx = positions[ci][0];
    const cy = positions[ci][1];
    let maxDist = -1;
    let contrastIdx = -1;
    for (let i = 0; i < N; i++) {
      if (i === ci) continue;
      if (!coffees[i].in_stock) continue;
      const dx = positions[i][0] - cx;
      const dy = positions[i][1] - cy;
      const d = dx * dx + dy * dy;
      if (d > maxDist) { maxDist = d; contrastIdx = i; }
    }
    contrasts.push(contrastIdx);
  }
  return contrasts;
}

// Apply archetype coloring for a given frame
function applyArchetypeColors(frameIdx) {
  if (frameIdx === currentFrameIdx) return;
  currentFrameIdx = frameIdx;
  const fa = frame_archetypes[frameIdx];

  // Update circle colors
  for (let i = 0; i < N; i++) {
    const archIdx = fa.dominant[i];
    const color = fa.archetypes[archIdx].color;
    circles[i].setAttribute('fill', color);
  }

  // Update legend
  rebuildLegend(fa);

  // Update arch count display
  archCountEl.textContent = fa.n_arch;
}

function rebuildLegend(fa) {
  legendEl.innerHTML = '';
  for (const arch of fa.archetypes) {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = '<div class="legend-swatch" style="background:' + arch.color + '"></div>' + arch.name;
    legendEl.appendChild(item);
  }
  const statusItems = [
    { label: '\\u2605 Archetype center', text: '\\u2605' },
    { label: '\\u25c6 Contrast (farthest)', text: '\\u25c6' },
    { label: 'In stock', swatch: 'background: #888; width: 9px; height: 9px;' },
    { label: 'Out of stock', swatch: 'background: #888; opacity: 0.3; width: 7px; height: 7px;' }
  ];
  for (const si of statusItems) {
    const item = document.createElement('div');
    item.className = 'legend-item';
    if (si.text) {
      item.innerHTML = '<span style="font-size:14px">' + si.text + '</span> ' + si.label;
    } else {
      item.innerHTML = '<div class="legend-swatch" style="' + si.swatch + '"></div>' + si.label;
    }
    legendEl.appendChild(item);
  }
}

function interpolate(weight) {
  currentWeight = weight;

  // Find bracketing frames
  let lo = 0, hi = nSteps - 1;
  for (let i = 0; i < nSteps - 1; i++) {
    if (steps[i + 1] >= weight) { lo = i; hi = i + 1; break; }
  }
  if (weight <= steps[0]) { lo = 0; hi = 0; }
  if (weight >= steps[nSteps - 1]) { lo = nSteps - 1; hi = nSteps - 1; }

  const t = (lo === hi) ? 0 : (weight - steps[lo]) / (steps[hi] - steps[lo]);
  const frameA = frames[lo];
  const frameB = frames[hi];

  // Snap archetype coloring to nearest frame
  const nearestFrame = (t <= 0.5) ? lo : hi;
  applyArchetypeColors(nearestFrame);
  const fa = frame_archetypes[nearestFrame];

  // Update positions and circles
  for (let i = 0; i < N; i++) {
    const x = frameA[i][0] * (1 - t) + frameB[i][0] * t;
    const y = frameA[i][1] * (1 - t) + frameB[i][1] * t;
    positions[i][0] = x;
    positions[i][1] = y;
    const [sx, sy] = toSVG(x, y);
    circles[i].setAttribute('cx', sx);
    circles[i].setAttribute('cy', sy);
  }

  // Update center stars and contrasts for current archetype set
  const contrasts = findContrasts(fa);
  for (let ai = 0; ai < maxArch; ai++) {
    if (ai < fa.n_arch) {
      const ci = fa.centers[ai];
      const color = fa.archetypes[ai].color;
      const [csx, csy] = toSVG(positions[ci][0], positions[ci][1]);

      centerStars[ai].setAttribute('x', csx);
      centerStars[ai].setAttribute('y', csy);
      centerStars[ai].setAttribute('fill', color);
      centerStars[ai].setAttribute('opacity', '1');

      const ri = contrasts[ai];
      if (ri >= 0) {
        const [rsx, rsy] = toSVG(positions[ri][0], positions[ri][1]);
        contrastDiamonds[ai].setAttribute('x', rsx);
        contrastDiamonds[ai].setAttribute('y', rsy);
        contrastDiamonds[ai].setAttribute('fill', color);
        contrastDiamonds[ai].setAttribute('opacity', '1');
        contrastLines[ai].setAttribute('x1', csx);
        contrastLines[ai].setAttribute('y1', csy);
        contrastLines[ai].setAttribute('x2', rsx);
        contrastLines[ai].setAttribute('y2', rsy);
        contrastLines[ai].setAttribute('stroke', color);
        contrastLines[ai].setAttribute('opacity', '0.6');
      } else {
        contrastDiamonds[ai].setAttribute('opacity', '0');
        contrastLines[ai].setAttribute('opacity', '0');
      }
    } else {
      // Hide unused archetype markers
      centerStars[ai].setAttribute('opacity', '0');
      contrastDiamonds[ai].setAttribute('opacity', '0');
      contrastLines[ai].setAttribute('opacity', '0');
    }
  }
}

// Slider event
slider.addEventListener('input', () => {
  const w = parseFloat(slider.value);
  weightDisplay.textContent = w.toFixed(2);
  interpolate(w);
});

// Tooltip with blended numeric/text descriptions
svg.addEventListener('mousemove', (e) => {
  const target = e.target;
  if (target.tagName === 'circle' && target.dataset.idx !== undefined) {
    const idx = parseInt(target.dataset.idx);
    const coffee = coffees[idx];
    const fa = frame_archetypes[currentFrameIdx];
    const archIdx = fa.dominant[idx];
    const arch = fa.archetypes[archIdx];
    const w = currentWeight;

    // Blend opacity: numeric fades out, text fades in
    const numOpacity = Math.max(0.3, 1.0 - w * 0.7);
    const txtOpacity = Math.max(0.3, w * 0.7 + 0.3);

    let descHtml = '';
    if (coffee.numeric) {
      descHtml += '<div class="tt-desc" style="opacity:' + numOpacity.toFixed(2) + '">'
        + '<span class="label">Cupping: </span>'
        + '<span class="value">' + coffee.numeric + '</span></div>';
    }
    if (coffee.text) {
      descHtml += '<div class="tt-desc" style="opacity:' + txtOpacity.toFixed(2) + '">'
        + '<span class="label">Notes: </span>'
        + '<span class="value">' + coffee.text + '</span></div>';
    }

    let status = coffee.in_stock ? 'In stock' : 'Out of stock';

    tooltip.innerHTML = '<div class="tt-name">' + coffee.name + '</div>'
      + '<div class="tt-arch">' + arch.name + '</div>'
      + descHtml
      + '<div class="tt-status">' + status + '</div>';
    tooltip.style.opacity = '1';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY + 14) + 'px';
  } else {
    tooltip.style.opacity = '0';
  }
});

svg.addEventListener('mouseleave', () => { tooltip.style.opacity = '0'; });

// Click to open URL
svg.addEventListener('click', (e) => {
  if (e.target.tagName === 'circle' && e.target.dataset.idx !== undefined) {
    const coffee = coffees[parseInt(e.target.dataset.idx)];
    if (coffee.url) window.open(coffee.url, '_blank');
  }
});

// Initial render
interpolate(0.0);
</script>
</body>
</html>"""
    )


# Flavor terms to ignore in keyword extraction (too generic for coffee context)
_TEXT_STOPWORDS = frozenset(
    [
        "coffee",
        "coffees",
        "cup",
        "cups",
        "roast",
        "roasts",
        "roasting",
        "roasted",
        "notes",
        "note",
        "flavor",
        "flavors",
        "aroma",
        "aromas",
        "aromatic",
        "aromatics",
        "city",
        "full",
        "light",
        "medium",
        "dark",
        "nice",
        "good",
        "great",
        "sweet",
        "sweetness",
        "bit",
        "hints",
        "hint",
        "like",
        "also",
        "well",
        "really",
        "quite",
        "overall",
        "makes",
        "made",
        "make",
        "will",
        "with",
        "that",
        "this",
        "from",
        "have",
        "some",
        "when",
        "than",
        "them",
        "they",
        "into",
        "been",
        "being",
        "were",
        "what",
        "there",
        "their",
        "about",
        "which",
        "would",
        "could",
        "should",
        "these",
        "those",
        "through",
        "does",
        "found",
        "here",
        "more",
        "most",
        "much",
        "other",
        "just",
        "over",
        "under",
        "after",
        "before",
        "between",
        "same",
        "still",
        "each",
        "both",
        "such",
        "only",
        "even",
        "back",
        "give",
        "gives",
        "come",
        "comes",
        "take",
        "takes",
        "keep",
        "keeps",
        "bring",
        "brings",
        "find",
        "finds",
        "show",
        "shows",
        "can",
        "one",
        "two",
        "best",
        "very",
        "accents",
        "accent",
        "tones",
        "tone",
        "range",
        "level",
        "slight",
        "slightly",
        "subtle",
        "strong",
        "stronger",
        "think",
        "profile",
        "profiles",
        "picked",
        "touch",
        "though",
        "little",
        "first",
        "second",
        "third",
        "think",
        "things",
        "thing",
        "something",
        "anything",
        "nothing",
        "everything",
        "along",
        "away",
        "down",
        "long",
        "high",
        "higher",
        "lower",
        "least",
        "last",
        "getting",
        "going",
        "want",
        "need",
        "much",
        "many",
        "while",
        "where",
        "across",
        "pulled",
        "around",
    ]
)


def _extract_keywords(text, top_n=8):
    """Extract distinctive flavor keywords from cupping notes text."""
    import re

    words = re.findall(r"[a-z]{4,}", text.lower())
    # Filter out generic stopwords
    words = [w for w in words if w not in _TEXT_STOPWORDS]
    # Count frequencies
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    # Sort by frequency, take top N
    ranked = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in ranked[:top_n]]


def _add_text_keywords(archetype_names, alpha, rows, n_arch, conn):
    """Enhance archetype names with distinctive noun phrases from purest members.

    Uses spaCy to extract noun phrases from cupping notes, then scores by
    lift (overrepresentation in archetype vs corpus) to find the most
    distinctive flavor descriptors for each archetype.
    """
    import numpy as np
    import spacy

    conn.row_factory = sqlite3.Row

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    # Get cupping notes for all coffees
    notes_by_url = {}
    for row in conn.execute(
        "SELECT url, cupping_notes FROM coffees WHERE cupping_notes IS NOT NULL"
    ).fetchall():
        notes_by_url[row["url"]] = row["cupping_notes"]

    def _extract_phrases(text):
        """Extract noun phrases from text via spaCy, lowercased and filtered."""
        doc = nlp(text)
        phrases = []
        # Non-flavor terms to filter out of phrases
        skip_words = {
            "opinion",
            "caffeine",
            "content",
            "works",
            "roasts",
            "roast",
            "city",
            "city+",
            "temperature",
            "batch",
            "brewing",
            "brewer",
            "grinder",
            "grounds",
            "water",
            "grams",
            "minutes",
            "seconds",
            "degrees",
            "ratio",
            "recommendation",
            "recommendations",
        }
        for chunk in doc.noun_chunks:
            # Strip determiners and pronouns from the phrase
            tokens = [t for t in chunk if t.pos_ not in ("DET", "PRON", "ADP")]
            if not tokens:
                continue
            phrase = " ".join(t.text.lower() for t in tokens)
            # Skip very short or very long phrases
            if len(phrase) < 4 or len(phrase) > 30:
                continue
            # Skip phrases with numbers or special chars (brewing instructions)
            if any(c.isdigit() or c in "()/+" for c in phrase):
                continue
            # Skip phrases containing non-flavor terms
            words = phrase.split()
            if any(w in skip_words for w in words):
                continue
            # Skip phrases that are entirely stopwords
            if all(w in _TEXT_STOPWORDS for w in words):
                continue
            # Skip proper nouns (farm names, place names)
            if all(t.pos_ == "PROPN" for t in tokens):
                continue
            phrases.append(phrase)
        return phrases

    # Compute corpus-wide phrase frequencies
    all_phrases = []
    for notes in notes_by_url.values():
        all_phrases.extend(_extract_phrases(notes))
    total_corpus = len(all_phrases)
    corpus_freq = {}
    for p in all_phrases:
        corpus_freq[p] = corpus_freq.get(p, 0) + 1

    enhanced_names = []
    for ai in range(n_arch):
        # Find top 5 purest members of this archetype
        dominance = alpha[:, ai]
        top_indices = np.argsort(dominance)[-5:][::-1]

        # Collect cupping notes from purest members
        arch_text = ""
        for idx in top_indices:
            url = rows[idx]["url"]
            if url in notes_by_url:
                arch_text += " " + notes_by_url[url]

        if not arch_text.strip():
            enhanced_names.append(archetype_names[ai])
            continue

        # Extract phrase frequencies for this archetype
        arch_phrases = _extract_phrases(arch_text)
        total_arch = len(arch_phrases)
        arch_freq = {}
        for p in arch_phrases:
            arch_freq[p] = arch_freq.get(p, 0) + 1

        # Score by lift, require minimum counts
        distinctive = []
        for phrase, count in arch_freq.items():
            if count < 2:
                continue
            corpus_count = corpus_freq.get(phrase, 0)
            if corpus_count < 2:
                continue
            arch_rate = count / total_arch
            corpus_rate = corpus_count / total_corpus
            lift = arch_rate / corpus_rate
            distinctive.append((phrase, lift))

        distinctive.sort(key=lambda x: -x[1])
        top_phrases = [phrase for phrase, _ in distinctive[:3]]

        if top_phrases:
            enhanced_names.append(f"{archetype_names[ai]} ({', '.join(top_phrases)})")
        else:
            enhanced_names.append(archetype_names[ai])

    return enhanced_names


def flavor_map(
    no_decaf=False,
    exclude_urls=None,
    available_only=False,
    variance_threshold=0.80,
    output_path="flavor-map.png",
    n_neighbors=10,
    min_dist=0.3,
    text_mode="none",
):
    """Generate a 2D UMAP flavor map as a PNG, colored by dominant archetype.

    text_mode controls how text embeddings influence the layout:
      - 'none': numeric PCA scores only (original behavior)
      - 'blended': precomputed distance matrix blending numeric + text cosine
      - 'concat': concatenate PCA-reduced text embeddings with numeric PCA scores
    """
    import numpy as np
    import umap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    conn = init_db()
    conn.row_factory = sqlite3.Row
    rows = get_scored_coffees(
        conn,
        no_decaf=no_decaf,
        exclude_urls=exclude_urls,
        available_only=available_only,
    )

    if len(rows) < 15:
        print("Need at least 15 scored coffees for UMAP projection.")
        conn.close()
        return

    # PCA reduction first (same foundation as archetypes)
    vecs = [to_vector(r) for r in rows]
    pca = pca_reduce(vecs, variance_threshold=variance_threshold)
    scores = np.array(pca["scores"])
    n_pca = pca["n_components"]

    # Load text embeddings if needed
    embeddings_dict = None
    if text_mode != "none":
        embeddings_dict = load_embeddings(conn)
        if not embeddings_dict:
            logger.warning(
                "text mode '%s' requested but no embeddings found, falling back to 'none'",
                text_mode,
            )
            text_mode = "none"

    # Build the space that drives UMAP and archetypal analysis
    # For concat, also build text components here so archetypes use the same space
    combined = None
    n_text_components = 0
    text_scores_scaled = None

    if text_mode == "concat":
        text_vecs = []
        for row in rows:
            emb = embeddings_dict.get(row["url"])
            if emb is not None:
                text_vecs.append(emb)
            else:
                text_vecs.append(np.zeros(384, dtype=np.float32))
        text_matrix = np.array(text_vecs, dtype=np.float64)

        # PCA reduce text embeddings
        text_mean = text_matrix.mean(axis=0)
        text_centered = text_matrix - text_mean
        U, S, Vt = np.linalg.svd(text_centered, full_matrices=False)
        text_var = (S**2) / (len(rows) - 1)
        text_cumvar = np.cumsum(text_var / text_var.sum())
        n_text_components = int(np.searchsorted(text_cumvar, 0.80) + 1)
        n_text_components = max(3, min(n_text_components, 10))
        text_scores = text_centered @ Vt[:n_text_components].T

        # Normalize both to similar scale before concatenating
        num_scale = np.std(scores)
        text_scale = np.std(text_scores)
        if text_scale > 0:
            text_scores_scaled = text_scores * (num_scale / text_scale)
        else:
            text_scores_scaled = text_scores

        combined = np.hstack([scores, text_scores_scaled])

    # Determine the archetype space: match what UMAP will use
    if text_mode == "concat":
        arch_input = combined
    else:
        arch_input = scores

    # Run archetypal analysis for coloring on the appropriate space
    n_arch, _ = _pick_n_archetypes(arch_input, seed=42)
    aa = archetypal_analysis(arch_input, n_arch, max_iter=300)
    alpha = np.array(aa["alpha"])
    arch_coords = np.array(aa["archetypes"])  # in arch_input space
    components = np.array(pca["components"])
    mean_22d = np.array(pca["mean"])

    # Name archetypes — project back to 22D for numeric labels
    ds = _dscale()
    stddevs = compute_dim_stddevs(vecs)

    if text_mode == "concat":
        # Archetype coords are in combined space: first n_pca dims are numeric PCA
        arch_numeric_scores = arch_coords[:, :n_pca]
    else:
        arch_numeric_scores = arch_coords

    arch_22d = arch_numeric_scores @ components + mean_22d
    archetype_names = []
    for ai in range(n_arch):
        arch_vec = arch_22d[ai]
        diffs = [(DIM_NAMES[j], (arch_vec[j] - mean_22d[j]) * ds[j]) for j in range(22)]
        diffs.sort(key=lambda x: -abs(x[1]))
        top_pos = [n for n, d in diffs if d > 0.3 * stddevs[0]][:2]
        archetype_names.append("/".join(top_pos) if top_pos else "Balanced")

    # Add text-derived keywords to archetype names when text is active
    if text_mode != "none" and embeddings_dict:
        archetype_names = _add_text_keywords(archetype_names, alpha, rows, n_arch, conn)

    # UMAP projection — varies by text_mode
    if text_mode == "blended":
        text_weight = _TEXT_WEIGHT if _TEXT_WEIGHT > 0 else 0.3
        logger.info(
            "running UMAP (blended): n_neighbors=%d, min_dist=%.2f, text_weight=%.2f",
            n_neighbors,
            min_dist,
            text_weight,
        )
        # Build pairwise distance matrix
        n = len(rows)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                nd = float(np.linalg.norm(scores[i] - scores[j]))
                emb_i = embeddings_dict.get(rows[i]["url"])
                emb_j = embeddings_dict.get(rows[j]["url"])
                if emb_i is not None and emb_j is not None:
                    td = 1.0 - float(emb_i @ emb_j)
                else:
                    td = nd
                blended = (1.0 - text_weight) * nd + text_weight * td
                dist_matrix[i, j] = blended
                dist_matrix[j, i] = blended

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="precomputed",
            random_state=42,
        )
        embedding = reducer.fit_transform(dist_matrix)
        umap_desc = f"UMAP(blended, text_weight={text_weight:.1f})"

    elif text_mode == "concat":
        logger.info(
            "running UMAP (concat): n_neighbors=%d, min_dist=%.2f, PCA+text → 2D",
            n_neighbors,
            min_dist,
        )
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="euclidean",
            random_state=42,
        )
        embedding = reducer.fit_transform(combined)
        umap_desc = f"UMAP(concat, {n_pca}D numeric + {n_text_components}D text → 2D)"

    else:  # none
        logger.info(
            "running UMAP: n_neighbors=%d, min_dist=%.2f, %dD PCA → 2D",
            n_neighbors,
            min_dist,
            n_pca,
        )
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="euclidean",
            random_state=42,
        )
        embedding = reducer.fit_transform(scores)
        umap_desc = f"UMAP({n_pca}D PCA → 2D)"

    # Classify each coffee
    dominant_arch = np.argmax(alpha, axis=1)

    # Color palette for archetypes
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / max(n_arch - 1, 1)) for i in range(n_arch)]

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")

    # Plot each coffee
    for i, row in enumerate(rows):
        x, y = embedding[i]
        arch_idx = dominant_arch[i]
        color = colors[arch_idx]
        in_stock = row["in_stock"]

        if in_stock:
            marker = "o"
            size = 60
            edge_color = "none"
            edge_width = 0
            zorder = 5
        else:
            marker = "x"
            size = 30
            edge_color = "none"
            edge_width = 0
            color = (*color[:3], 0.3)
            zorder = 2

        scatter_kwargs = dict(
            c=[color],
            marker=marker,
            s=size,
            zorder=zorder,
        )
        if marker != "x":
            scatter_kwargs["edgecolors"] = edge_color
            scatter_kwargs["linewidths"] = edge_width

        ax.scatter(x, y, **scatter_kwargs)

    # Label purest archetype expressions
    labeled_indices = set()

    # Find archetype centers (purest expression) and contrasts, highlight them
    arch_center_indices = []
    arch_contrast_indices = []
    avail_indices = [i for i, row in enumerate(rows) if row["in_stock"]]

    for ai in range(n_arch):
        # Purest expression: highest alpha for this archetype (prefer in-stock)
        best_idx = None
        best_w = 0
        for i in avail_indices:
            if alpha[i, ai] > best_w:
                best_w = alpha[i, ai]
                best_idx = i
        if best_idx is None:
            # Fall back to any coffee
            best_idx = int(np.argmax(alpha[:, ai]))

        arch_center_indices.append(best_idx)
        labeled_indices.add(best_idx)

        # Contrast: coffee farthest from this center in 2D embedding space
        center_xy = embedding[best_idx]
        contrast_idx = None
        max_dist = -1
        for i in avail_indices:
            if i == best_idx:
                continue
            d = float(np.linalg.norm(embedding[i] - center_xy))
            if d > max_dist:
                max_dist = d
                contrast_idx = i
        arch_contrast_indices.append(contrast_idx)
        if contrast_idx is not None:
            labeled_indices.add(contrast_idx)

    # Draw lines between center and contrast for each archetype
    for ai in range(n_arch):
        center_idx = arch_center_indices[ai]
        contrast_idx = arch_contrast_indices[ai]
        if contrast_idx is None:
            continue
        cx, cy = embedding[center_idx]
        rx, ry = embedding[contrast_idx]
        ax.plot(
            [cx, rx],
            [cy, ry],
            color=colors[ai],
            linewidth=1.2,
            alpha=0.5,
            linestyle="--",
            zorder=8,
        )

    # Draw highlighted star markers on archetype centers
    for ai in range(n_arch):
        center_idx = arch_center_indices[ai]
        cx, cy = embedding[center_idx]
        # Outer glow
        ax.scatter(
            cx,
            cy,
            c=[colors[ai]],
            marker="*",
            s=400,
            alpha=0.3,
            zorder=11,
            edgecolors="none",
        )
        # Star marker
        ax.scatter(
            cx,
            cy,
            c=[colors[ai]],
            marker="*",
            s=200,
            zorder=12,
            edgecolors="white",
            linewidths=0.8,
        )

    # Draw diamond on contrast coffees
    for ai in range(n_arch):
        contrast_idx = arch_contrast_indices[ai]
        if contrast_idx is None:
            continue
        rx, ry = embedding[contrast_idx]
        ax.scatter(
            rx,
            ry,
            c=[colors[ai]],
            marker="D",
            s=100,
            zorder=11,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.8,
        )

    for i in labeled_indices:
        x, y = embedding[i]
        name = rows[i]["name"]
        # Truncate long names
        if len(name) > 25:
            name = name[:23] + "…"
        ax.annotate(
            name,
            (x, y),
            fontsize=6.5,
            color="white",
            alpha=0.85,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Legend
    legend_elements = []
    for ai in range(n_arch):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[ai],
                markersize=10,
                label=archetype_names[ai],
                linestyle="None",
            )
        )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="In stock",
            linestyle="None",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="x",
            color="gray",
            markersize=8,
            label="Out of stock",
            linestyle="None",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markeredgecolor="white",
            markersize=12,
            label="Archetype center",
            linestyle="None",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="white",
            markersize=8,
            label="Contrast (farthest)",
            linestyle="None",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.2,
            linestyle="--",
            alpha=0.6,
            label="Center ↔ Contrast",
        )
    )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=8,
        facecolor="#2a2a4a",
        edgecolor="gray",
        labelcolor="white",
        framealpha=0.9,
    )

    ax.set_title(
        f"Coffee Flavor Map — {len(rows)} coffees, {n_arch} archetypes, {umap_desc}",
        color="white",
        fontsize=12,
        pad=15,
    )
    ax.set_xlabel("UMAP 1", color="white", fontsize=9)
    ax.set_ylabel("UMAP 2", color="white", fontsize=9)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("gray")

    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05)
    dpi = 150
    plt.savefig(output_path, dpi=dpi)

    # Compute pixel coordinates for HTML overlay
    # Use fig.dpi (internal) not save dpi — transData works in internal coords
    internal_w = fig.get_size_inches()[0] * fig.dpi
    internal_h = fig.get_size_inches()[1] * fig.dpi
    hotspots = []
    for i, row in enumerate(rows):
        if not row["in_stock"]:
            continue
        display_coords = ax.transData.transform(embedding[i])
        pct_x = display_coords[0] / internal_w * 100
        pct_y = (internal_h - display_coords[1]) / internal_h * 100
        arch_idx = dominant_arch[i]
        hotspots.append(
            {
                "name": row["name"],
                "url": row["url"],
                "x": pct_x,
                "y": pct_y,
                "archetype": archetype_names[arch_idx],
            }
        )

    plt.close()

    # Generate HTML file
    import os

    html_path = os.path.splitext(output_path)[0] + ".html"
    png_basename = os.path.basename(output_path)
    _write_map_html(html_path, png_basename, hotspots)

    print(f"\n  Saved: {output_path}")
    print(f"  Saved: {html_path}")
    print(f"  {len(rows)} coffees, {n_arch} archetypes, {umap_desc}")
    print(f"  UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}")
    print(f"  Text mode: {text_mode}")
    print(f"  {len(hotspots)} clickable hotspots (in-stock)")

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
    common.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="URL_OR_KEYWORD",
        help="exclude a coffee from results (repeatable, URL or name keyword)",
    )
    common.add_argument(
        "--offline",
        action="store_true",
        help="skip all fetching (use existing DB only)",
    )
    common.add_argument(
        "--no-zscore",
        action="store_true",
        help="disable z-score normalization (use raw value/scale)",
    )
    common.add_argument(
        "--distance",
        choices=["l2", "l1", "cosine", "mahalanobis"],
        default="l2",
        metavar="METRIC",
        help="distance function: l2, l1, cosine, mahalanobis (default: %(default)s)",
    )
    common.add_argument(
        "--pca-variance",
        type=float,
        default=0.80,
        metavar="FRAC",
        help="PCA variance threshold for auto-selecting components (default: %(default)s)",
    )
    common.add_argument(
        "--text-weight",
        type=float,
        default=0.0,
        metavar="W",
        help="blend text embedding similarity into distance (0.0–1.0, default: %(default)s). "
        "Requires embeddings: run 'python embed.py build' first.",
    )

    parser = argparse.ArgumentParser(
        prog="coffee.py",
        description="Coffee flavor space analysis",
        epilog=f"Database: {DB_PATH}",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- recommend ---
    p = sub.add_parser(
        "recommend",
        parents=[common],
        help="recommend next coffee (max variation from tried)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=3,
        metavar="N",
        help="number of top recommendations (default: %(default)s)",
    )

    # --- profile ---
    p = sub.add_parser(
        "profile",
        parents=[common],
        help="profile a coffee: dimensions, archetype mix, contrast pair, clusters",
    )
    p.add_argument("query", help="coffee name (fuzzy) or URL")

    # --- explore ---
    sub.add_parser(
        "explore",
        parents=[common],
        help="find high/low pairs per dimension to isolate flavors",
    )

    # --- pairs ---
    p = sub.add_parser(
        "pairs",
        parents=[common],
        help="assign coffees into tasting contrast pairs by flavor similarity",
    )
    p.add_argument(
        "coffees",
        nargs="+",
        help="coffee names (fuzzy) or URLs to pair up",
    )

    # --- factors ---
    p = sub.add_parser(
        "factors",
        parents=[common],
        help="PCA factor analysis: latent flavor themes and exploration gaps",
    )
    p.add_argument(
        "--components",
        type=int,
        default=None,
        metavar="N",
        help="force N components (default: auto via --pca-variance threshold)",
    )
    p.add_argument(
        "--available",
        action="store_true",
        help="only analyze currently in-stock coffees",
    )

    # --- archetypes ---
    p = sub.add_parser(
        "archetypes",
        parents=[common],
        help="archetypal analysis: extreme styles and mixture decomposition",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        metavar="N",
        help="number of archetypes (default: auto-select 3–7)",
    )
    p.add_argument(
        "--available",
        action="store_true",
        help="only analyze currently in-stock coffees",
    )

    # --- map ---
    p = sub.add_parser(
        "map",
        parents=[common],
        help="generate 2D UMAP flavor map as PNG",
    )
    p.add_argument(
        "--output",
        default="flavor-map.png",
        metavar="PATH",
        help="output PNG path (default: %(default)s)",
    )
    p.add_argument(
        "--neighbors",
        type=int,
        default=10,
        metavar="N",
        help="UMAP n_neighbors param — lower=more local structure (default: %(default)s)",
    )
    p.add_argument(
        "--min-dist",
        type=float,
        default=0.3,
        metavar="D",
        help="UMAP min_dist param — lower=tighter clusters (default: %(default)s)",
    )
    p.add_argument(
        "--available",
        action="store_true",
        help="only map currently in-stock coffees",
    )
    p.add_argument(
        "--text-mode",
        choices=["none", "blended", "concat"],
        default="none",
        metavar="MODE",
        help="how text embeddings influence layout: none (numeric only), "
        "blended (precomputed hybrid distance), concat (concatenate reduced "
        "text+numeric vectors). Requires embeddings: run 'python embed.py build' "
        "first. (default: %(default)s)",
    )
    p.add_argument(
        "--slider",
        action="store_true",
        help="generate interactive HTML with text-weight slider (0→1). "
        "Pre-computes UMAP at multiple steps with Procrustes alignment.",
    )
    p.add_argument(
        "--slider-steps",
        type=int,
        default=11,
        metavar="N",
        help="number of weight steps for slider map (default: %(default)s)",
    )

    # --- insights ---
    p = sub.add_parser(
        "insights",
        parents=[common],
        help="collection analysis: outliers, clusters, superlatives",
    )
    p.add_argument(
        "--clusters-only",
        action="store_true",
        help="show only the k-means cluster analysis",
    )
    p.add_argument(
        "--available",
        action="store_true",
        help="only analyze currently in-stock coffees",
    )

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
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    no_decaf = not args.decaf
    _DISTANCE_METRIC = args.distance
    _TEXT_WEIGHT = args.text_weight
    use_zscore = not args.no_zscore

    conn = sqlite3.connect(DB_PATH)
    if use_zscore:
        init_zscore(conn)
    ensure_embeddings(conn)
    _init_text_embeddings(conn)
    conn.close()

    if args.command == "recommend":
        recommend(
            no_decaf=no_decaf,
            exclude_urls=args.exclude,
            offline=args.offline,
            top_n=args.top,
        )
    elif args.command == "profile":
        profile(args.query, no_decaf=no_decaf)
    elif args.command == "explore":
        explore(no_decaf=no_decaf, exclude_urls=args.exclude)
    elif args.command == "pairs":
        pairs(args.coffees)
    elif args.command == "factors":
        factors(
            no_decaf=no_decaf,
            exclude_urls=args.exclude,
            n_components=args.components,
            available_only=args.available,
            variance_threshold=args.pca_variance,
        )
    elif args.command == "archetypes":
        archetypes(
            no_decaf=no_decaf,
            exclude_urls=args.exclude,
            n_archetypes=args.k,
            available_only=args.available,
            variance_threshold=args.pca_variance,
        )
    elif args.command == "map":
        if args.slider:
            import os

            slider_output = os.path.splitext(args.output)[0] + "-slider.html"
            flavor_map_slider(
                no_decaf=no_decaf,
                exclude_urls=args.exclude,
                available_only=args.available,
                variance_threshold=args.pca_variance,
                output_path=slider_output,
                n_neighbors=args.neighbors,
                min_dist=args.min_dist,
                n_steps=args.slider_steps,
            )
        else:
            flavor_map(
                no_decaf=no_decaf,
                exclude_urls=args.exclude,
                available_only=args.available,
                variance_threshold=args.pca_variance,
                output_path=args.output,
                n_neighbors=args.neighbors,
                min_dist=args.min_dist,
                text_mode=args.text_mode,
            )
    elif args.command == "insights":
        insights(
            no_decaf=no_decaf,
            exclude_urls=args.exclude,
            clusters_only=args.clusters_only,
            available_only=args.available,
        )
