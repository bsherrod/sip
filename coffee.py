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
    """Return a distance function based on the selected metric and the population vectors."""
    metric = _DISTANCE_METRIC

    if metric == "l1":

        def distance(a, b):
            return sum(abs(x - y) for x, y in zip(a, b))

    elif metric == "cosine":

        def distance(a, b):
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

        def distance(a, b):
            diff = np.array(a) - np.array(b)
            return float(np.sqrt(diff @ cov_inv @ diff))

    else:  # l2 (default)
        weights = compute_dim_weights(all_vecs)

        def distance(a, b):
            return sum(w * (x - y) ** 2 for x, y, w in zip(a, b, weights)) ** 0.5

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
    tried_vectors = [(to_vector(r), r["rating"]) for r in tried_for_distance]
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
        for tvec, rating in tried_vectors:
            d = weighted_distance(vec, tvec) * rating_weights.get(rating, 1.0)
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


def compare(query, no_decaf=False):
    """Compare a single coffee to tried coffees and place it in k-means clusters."""
    random.seed(42)

    conn = init_db()
    conn.row_factory = sqlite3.Row

    target = find_coffee(conn, query, "compare")
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
        d = weighted_distance(target_vec, tvec)
        tried_dists.append((tr, d, tvec))
    tried_dists.sort(key=lambda x: x[1])

    stock_str = "In Stock ✅" if target["in_stock"] else "Out of Stock ❌"
    print(f"\n{'━' * 60}")
    print(f"  COMPARE: {target['name']}")
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
            (c, weighted_distance(to_vector(c), target_vec)) for c in candidates
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


def flavor_map(
    no_decaf=False,
    exclude_urls=None,
    available_only=False,
    variance_threshold=0.80,
    output_path="flavor-map.png",
    n_neighbors=10,
    min_dist=0.3,
):
    """Generate a 2D UMAP flavor map as a PNG, colored by dominant archetype."""
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

    # Run archetypal analysis for coloring
    n_arch, _ = _pick_n_archetypes(scores, seed=42)
    aa = archetypal_analysis(scores, n_arch, max_iter=300)
    alpha = np.array(aa["alpha"])
    arch_scores = np.array(aa["archetypes"])
    components = np.array(pca["components"])
    mean_22d = np.array(pca["mean"])

    # Name archetypes (same logic as archetypes command)
    arch_22d = arch_scores @ components + mean_22d
    ds = _dscale()
    stddevs = compute_dim_stddevs(vecs)
    archetype_names = []
    for ai in range(n_arch):
        arch_vec = arch_22d[ai]
        diffs = [(DIM_NAMES[j], (arch_vec[j] - mean_22d[j]) * ds[j]) for j in range(22)]
        diffs.sort(key=lambda x: -abs(x[1]))
        top_pos = [n for n, d in diffs if d > 0.3 * stddevs[0]][:2]
        archetype_names.append("/".join(top_pos) if top_pos else "Balanced")

    # UMAP projection
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

    # Classify each coffee
    tried_urls = set(r[0] for r in conn.execute("SELECT url FROM tried").fetchall())
    tried_ratings = dict(conn.execute("SELECT url, rating FROM tried").fetchall())

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
        is_tried = row["url"] in tried_urls
        in_stock = row["in_stock"]

        if is_tried:
            rating = tried_ratings.get(row["url"], "0")
            marker = {"+": "^", "0": "s", "-": "v"}.get(rating, "o")
            size = 120
            edge_color = "white"
            edge_width = 1.5
            zorder = 10
        elif in_stock:
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

    # Label tried coffees and purest archetype expressions
    labeled_indices = set()

    # Label all tried coffees
    for i, row in enumerate(rows):
        if row["url"] in tried_urls:
            labeled_indices.add(i)

    # Label purest expression of each archetype (in-stock untried)
    for ai in range(n_arch):
        best_idx = None
        best_w = 0
        for i, row in enumerate(rows):
            if (
                row["in_stock"]
                and row["url"] not in tried_urls
                and alpha[i, ai] > best_w
            ):
                best_w = alpha[i, ai]
                best_idx = i
        if best_idx is not None:
            labeled_indices.add(best_idx)

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
            marker="^",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="white",
            markersize=10,
            label="Tried (liked)",
            linestyle="None",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="white",
            markersize=10,
            label="Tried (neutral)",
            linestyle="None",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="white",
            markersize=10,
            label="Tried (disliked)",
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
            label="Untried (in stock)",
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
        f"Coffee Flavor Map — {len(rows)} coffees, {n_arch} archetypes, "
        f"UMAP({n_pca}D PCA → 2D)",
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
        if not row["in_stock"] or row["url"] in tried_urls:
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
    print(f"  {len(rows)} coffees, {n_arch} archetypes, {n_pca}D PCA → 2D UMAP")
    print(f"  UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}")
    print(f"  {len(hotspots)} clickable hotspots (in-stock, untried)")

    # Text summary: spatial observations
    tried_indices = [i for i, r in enumerate(rows) if r["url"] in tried_urls]
    untried_avail = [
        i for i, r in enumerate(rows) if r["url"] not in tried_urls and r["in_stock"]
    ]

    if tried_indices and untried_avail:
        tried_emb = embedding[tried_indices]
        tried_center = tried_emb.mean(axis=0)
        tried_radius = np.max(np.linalg.norm(tried_emb - tried_center, axis=1))

        # Find untried coffees farthest from tried center
        untried_dists = [
            (i, float(np.linalg.norm(embedding[i] - tried_center)))
            for i in untried_avail
        ]
        untried_dists.sort(key=lambda x: -x[1])

        print("\n  ── Spatial Blind Spots ──")
        print(f"  Your tried coffees span a radius of {tried_radius:.2f} in 2D space.")
        print("  Farthest available coffees from your explored region:\n")
        for idx, dist in untried_dists[:5]:
            coffee = rows[idx]
            arch_idx = dominant_arch[idx]
            mix = alpha[idx]
            top_arch = sorted(
                [(archetype_names[a], mix[a]) for a in range(n_arch)],
                key=lambda x: -x[1],
            )
            parts_str = " + ".join(f"{n} {w:.0%}" for n, w in top_arch if w > 0.1)
            beyond = "⚡" if dist > tried_radius else " "
            print(f"    {beyond} {coffee['name']} (dist={dist:.2f})")
            print(f"       {coffee['url']}")
            print(f"       {parts_str}")

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

    # --- compare ---
    p = sub.add_parser(
        "compare",
        parents=[common],
        help="compare a coffee to tried & place in clusters",
    )
    p.add_argument("query", help="coffee name (fuzzy) or URL")

    # --- explore ---
    sub.add_parser(
        "explore",
        parents=[common],
        help="find high/low pairs per dimension to isolate flavors",
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
        level=log_level, format="%(levelname)s: %(message)s", stream=sys.stderr
    )

    no_decaf = not args.decaf
    _DISTANCE_METRIC = args.distance
    use_zscore = not args.no_zscore

    if use_zscore:
        init_zscore(sqlite3.connect(DB_PATH))

    if args.command == "recommend":
        recommend(
            no_decaf=no_decaf,
            exclude_urls=args.exclude,
            offline=args.offline,
            top_n=args.top,
        )
    elif args.command == "compare":
        compare(args.query, no_decaf=no_decaf)
    elif args.command == "explore":
        explore(no_decaf=no_decaf, exclude_urls=args.exclude)
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
        flavor_map(
            no_decaf=no_decaf,
            exclude_urls=args.exclude,
            available_only=args.available,
            variance_threshold=args.pca_variance,
            output_path=args.output,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
        )
    elif args.command == "insights":
        insights(
            no_decaf=no_decaf,
            exclude_urls=args.exclude,
            clusters_only=args.clusters_only,
            available_only=args.available,
        )
