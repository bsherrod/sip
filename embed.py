#!/usr/bin/env python3
"""Compute and manage sentence embeddings for coffee cupping notes.

Embeds cupping_notes text from the coffees database using sentence-transformers,
stores results in a SQLite table, and provides loading utilities for use in
distance calculations.

Usage:
    python embed.py build       # compute embeddings for all coffees with notes
    python embed.py build --force  # recompute all (ignore existing)
    python embed.py status      # show how many embeddings exist
"""

import argparse
import logging
import sqlite3
import struct
import sys

import numpy as np

from scrape_sm import init_db

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _init_embeddings_table(conn):
    """Create the embeddings table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            url TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            vector BLOB NOT NULL,
            text_hash TEXT NOT NULL
        )
    """)
    conn.commit()


def _text_hash(text):
    """Short hash of text to detect when notes change and need re-embedding."""
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _pack_vector(vec):
    """Pack a numpy float32 array into bytes for SQLite BLOB storage."""
    return struct.pack(f"{len(vec)}f", *vec.astype(np.float32))


def _unpack_vector(blob):
    """Unpack a BLOB back into a numpy float32 array."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def load_embeddings(conn):
    """Load all embeddings from the database.

    Returns dict mapping url -> numpy float32 array (384D).
    Returns empty dict if no embeddings exist.
    """
    _init_embeddings_table(conn)
    rows = conn.execute("SELECT url, vector FROM embeddings").fetchall()
    return {url: _unpack_vector(blob) for url, blob in rows}


def build_embeddings(conn, force=False):
    """Compute embeddings for all coffees with cupping_notes.

    Skips coffees that already have an up-to-date embedding (same text hash)
    unless force=True.
    """
    from sentence_transformers import SentenceTransformer

    _init_embeddings_table(conn)
    conn.row_factory = sqlite3.Row

    coffees = conn.execute(
        "SELECT url, cupping_notes FROM coffees WHERE cupping_notes IS NOT NULL"
    ).fetchall()

    if not coffees:
        logger.warning("no coffees with cupping_notes found in database")
        return 0

    # Determine which need (re)computing
    existing = {}
    if not force:
        rows = conn.execute("SELECT url, text_hash FROM embeddings").fetchall()
        existing = {r["url"]: r["text_hash"] for r in rows}

    to_embed = []
    for coffee in coffees:
        current_hash = _text_hash(coffee["cupping_notes"])
        if not force and existing.get(coffee["url"]) == current_hash:
            continue
        to_embed.append((coffee["url"], coffee["cupping_notes"], current_hash))

    if not to_embed:
        logger.info("all embeddings up to date, nothing to compute")
        return 0

    logger.info("loading model %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    texts = [t[1] for t in to_embed]
    logger.info("encoding %d texts", len(texts))
    vectors = model.encode(
        texts, show_progress_bar=len(texts) > 50, normalize_embeddings=True
    )

    for i, (url, _text, text_h) in enumerate(to_embed):
        blob = _pack_vector(vectors[i])
        conn.execute(
            """INSERT OR REPLACE INTO embeddings (url, model, vector, text_hash)
            VALUES (?, ?, ?, ?)""",
            (url, MODEL_NAME, blob, text_h),
        )

    conn.commit()
    logger.info("stored %d embeddings", len(to_embed))
    return len(to_embed)


def status(conn):
    """Print embedding status."""
    _init_embeddings_table(conn)
    total_coffees = conn.execute(
        "SELECT COUNT(*) FROM coffees WHERE cupping_notes IS NOT NULL"
    ).fetchone()[0]
    total_embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    stale = 0
    if total_embeddings > 0:
        rows = conn.execute(
            """SELECT e.url, e.text_hash, c.cupping_notes
            FROM embeddings e JOIN coffees c ON e.url = c.url
            WHERE c.cupping_notes IS NOT NULL"""
        ).fetchall()
        for r in rows:
            if _text_hash(r[2]) != r[1]:
                stale += 1

    print(f"  Coffees with cupping notes: {total_coffees}")
    print(f"  Embeddings stored:          {total_embeddings}")
    print(f"  Missing:                    {total_coffees - total_embeddings}")
    print(f"  Stale (text changed):       {stale}")
    print(f"  Model:                      {MODEL_NAME} ({EMBEDDING_DIM}D)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="embed.py", description="Manage sentence embeddings for cupping notes"
    )
    parser.add_argument("--verbose", action="store_true", help="show debug logging")
    parser.add_argument("--quiet", action="store_true", help="suppress info logging")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    p = sub.add_parser("build", help="compute embeddings for all cupping notes")
    p.add_argument(
        "--force", action="store_true", help="recompute all embeddings even if current"
    )
    sub.add_parser("status", help="show embedding coverage")

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

    conn = init_db()
    if args.command == "build":
        n = build_embeddings(conn, force=args.force)
        print(f"  Computed {n} embeddings.")
    elif args.command == "status":
        status(conn)
    conn.close()
