"""
meta_crawler.py — Lightweight Metadata Crawler for NexusHub
-------------------------------------------------------------
Crawls ONLY metadata for ghost nodes (papers referenced in citations
but never fully crawled). No abstract needed — just enough to make
the citation graph dense and analysable.

Fetches per paper:
  - title, year, citation_count
  - top concepts (for bridge detection)
  - journal name

Does NOT fetch:
  - abstract (skip, saves 80% of data)
  - full author list
  - references (no recursive expansion)
  - related works

This turns 900k ghost nodes into real nodes with metadata,
making the citation graph analysable without full crawling.

Usage:
    python meta_crawler.py              # crawl all ghost nodes
    python meta_crawler.py --limit 50000  # crawl first 50k
    python meta_crawler.py --stats      # just show stats
"""

import sqlite3
import asyncio
import aiohttp
import argparse
import time
import json

DB_PATH = "knowledge.db"
MAX_CONCURRENT = 20
BATCH_SIZE     = 100
RATE_SLEEP     = 0.05   # 50ms between requests

# ── DB ────────────────────────────────────────────────────────────────────────

def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def ensure_schema(conn):
    """Add meta_only flag to papers table if not exists."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id       TEXT PRIMARY KEY,
            title          TEXT,
            doi            TEXT,
            year           INTEGER,
            journal        TEXT,
            citation_count INTEGER,
            abstract       TEXT
        );

        CREATE TABLE IF NOT EXISTS concepts (
            concept_id TEXT PRIMARY KEY,
            name       TEXT,
            level      INTEGER,
            wikidata   TEXT
        );

        CREATE TABLE IF NOT EXISTS paper_concepts (
            paper_id   TEXT,
            concept_id TEXT,
            score      REAL
        );

        CREATE TABLE IF NOT EXISTS meta_crawl_queue (
            paper_id   TEXT PRIMARY KEY,
            status     TEXT DEFAULT 'pending',
            attempts   INTEGER DEFAULT 0
        );
    """)
    conn.commit()

def print_stats(conn):
    print("\nDATABASE STATS")
    print("-" * 50)

    total_papers  = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    with_abstract = conn.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL").fetchone()[0]
    meta_only     = total_papers - with_abstract
    total_cites   = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]

    ghost = conn.execute("""
        SELECT COUNT(DISTINCT cited_paper_id) FROM citations
        WHERE cited_paper_id NOT IN (SELECT paper_id FROM papers)
    """).fetchone()[0]

    real_edges = conn.execute("""
        SELECT COUNT(*) FROM citations c
        WHERE EXISTS (SELECT 1 FROM papers WHERE paper_id = c.paper_id)
          AND EXISTS (SELECT 1 FROM papers WHERE paper_id = c.cited_paper_id)
    """).fetchone()[0]

    queued = conn.execute(
        "SELECT COUNT(*) FROM meta_crawl_queue WHERE status='pending'"
    ).fetchone()[0]

    done = conn.execute(
        "SELECT COUNT(*) FROM meta_crawl_queue WHERE status='done'"
    ).fetchone()[0]

    print(f"  Full papers (with abstract): {with_abstract:,}")
    print(f"  Meta-only papers:            {meta_only:,}")
    print(f"  Ghost nodes (not in DB yet): {ghost:,}")
    print(f"  Total citation edges:        {total_cites:,}")
    print(f"  Real edges (both in DB):     {real_edges:,}")
    print(f"  Queue pending:               {queued:,}")
    print(f"  Queue done:                  {done:,}")

    if total_cites > 0:
        coverage = real_edges / total_cites * 100
        print(f"  Citation coverage:           {coverage:.1f}%")
    print()

# ── QUEUE BUILDER ─────────────────────────────────────────────────────────────

def build_queue(conn, limit=None):
    print("Building meta-crawl queue from ghost nodes...")

    # Find all paper IDs referenced in citations but not in papers table
    ghost_ids = conn.execute("""
        SELECT DISTINCT cited_paper_id as pid FROM citations
        WHERE cited_paper_id NOT IN (SELECT paper_id FROM papers)
        UNION
        SELECT DISTINCT paper_id as pid FROM citations
        WHERE paper_id NOT IN (SELECT paper_id FROM papers)
    """).fetchall()

    ghost_set = [r["pid"] for r in ghost_ids]

    if limit:
        # Prioritize by how many times they appear in citations
        # (most cited ghost nodes first — these are most important)
        freq = conn.execute("""
            SELECT cited_paper_id as pid, COUNT(*) as cnt
            FROM citations
            WHERE cited_paper_id NOT IN (SELECT paper_id FROM papers)
            GROUP BY cited_paper_id
            ORDER BY cnt DESC
            LIMIT ?
        """, (limit,)).fetchall()
        ghost_set = [r["pid"] for r in freq]
        print(f"  Prioritized top {limit:,} most-cited ghost nodes")
    else:
        print(f"  Found {len(ghost_set):,} ghost nodes")

    added = 0
    for pid in ghost_set:
        conn.execute(
            "INSERT OR IGNORE INTO meta_crawl_queue VALUES (?, 'pending', 0)",
            (pid,)
        )
        added += 1

    conn.commit()
    print(f"  Added {added:,} to queue")
    return added

# ── FETCH ─────────────────────────────────────────────────────────────────────

async def fetch_meta(session, pid):
    """Fetch minimal metadata for a single paper."""
    url = f"https://api.openalex.org/works/{pid}"

    # Only request fields we need — much faster than full fetch
    fields = "id,title,publication_year,cited_by_count,primary_location,concepts"
    url += f"?select={fields}"

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status == 404:
                return pid, None   # paper doesn't exist, skip
            if r.status != 200:
                return pid, "error"
            data = await r.json()
            if not data.get("id"):
                return pid, None
            return pid, data
    except Exception as e:
        return pid, "error"

# ── STORE ─────────────────────────────────────────────────────────────────────

def store_meta(conn, pid, data):
    """Store minimal metadata — no abstract, no authors, no references."""
    if not data:
        return

    title = data.get("title")
    year  = data.get("publication_year")
    cites = data.get("cited_by_count", 0)

    journal = None
    primary = data.get("primary_location") or {}
    source  = primary.get("source") if isinstance(primary, dict) else None
    if source:
        journal = source.get("display_name")

    # Insert paper WITHOUT abstract (meta-only)
    conn.execute("""
        INSERT OR IGNORE INTO papers
        (paper_id, title, doi, year, journal, citation_count, abstract)
        VALUES (?, ?, NULL, ?, ?, ?, NULL)
    """, (pid, title, year, journal, cites))

    # Store top 3 concepts only (enough for bridge detection)
    concepts = sorted(
        data.get("concepts", []),
        key=lambda c: c.get("score", 0),
        reverse=True
    )[:3]

    for c in concepts:
        cid = c.get("id", "").split("/")[-1]
        if not cid:
            continue
        name     = c.get("display_name")
        level    = c.get("level")
        wikidata = c.get("wikidata")
        score    = c.get("score")

        conn.execute(
            "INSERT OR IGNORE INTO concepts VALUES (?, ?, ?, ?)",
            (cid, name, level, wikidata)
        )
        conn.execute(
            "INSERT OR IGNORE INTO paper_concepts VALUES (?, ?, ?)",
            (pid, cid, score)
        )

# ── BATCH PROCESSOR ───────────────────────────────────────────────────────────

async def process_batch(session, batch):
    tasks = [fetch_meta(session, pid) for pid in batch]
    return await asyncio.gather(*tasks, return_exceptions=True)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

async def run(db_path, limit):
    conn = connect(db_path)
    ensure_schema(conn)

    print_stats(conn)

    # Build queue
    pending = conn.execute(
        "SELECT COUNT(*) FROM meta_crawl_queue WHERE status='pending'"
    ).fetchone()[0]

    if pending == 0:
        build_queue(conn, limit)

    # Count total work
    total = conn.execute(
        "SELECT COUNT(*) FROM meta_crawl_queue WHERE status='pending'"
    ).fetchone()[0]

    print(f"Starting meta-crawl: {total:,} papers to process")
    print(f"Concurrent requests: {MAX_CONCURRENT}")
    print(f"Estimated time: ~{total/MAX_CONCURRENT*0.1/60:.0f} minutes\n")

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    headers   = {"User-Agent": "NexusHub/1.0 (research project)"}

    done  = 0
    errors= 0
    start = time.time()

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        while True:
            # Load batch
            rows = conn.execute("""
                SELECT paper_id FROM meta_crawl_queue
                WHERE status = 'pending'
                LIMIT ?
            """, (BATCH_SIZE,)).fetchall()

            if not rows:
                break

            batch = [r["paper_id"] for r in rows]

            # Mark as processing
            for pid in batch:
                conn.execute(
                    "UPDATE meta_crawl_queue SET status='processing' WHERE paper_id=?",
                    (pid,)
                )
            conn.commit()

            # Fetch all in parallel
            results = await process_batch(session, batch)

            # Store results
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                    continue

                pid, data = result

                if data == "error":
                    # Put back in queue for retry (up to 2 attempts)
                    conn.execute("""
                        UPDATE meta_crawl_queue
                        SET status = CASE WHEN attempts >= 2 THEN 'failed' ELSE 'pending' END,
                            attempts = attempts + 1
                        WHERE paper_id = ?
                    """, (pid,))
                    errors += 1
                else:
                    store_meta(conn, pid, data)
                    conn.execute(
                        "UPDATE meta_crawl_queue SET status='done' WHERE paper_id=?",
                        (pid,)
                    )
                    done += 1

            conn.commit()

            # Progress
            elapsed = time.time() - start
            rate    = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate / 60 if rate > 0 else 0

            print(f"  [{done:,}/{total:,}] "
                  f"rate: {rate:.0f}/s  "
                  f"errors: {errors}  "
                  f"eta: {remaining:.0f}m")

            await asyncio.sleep(RATE_SLEEP)

    conn.commit()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"  Stored:  {done:,} meta papers")
    print(f"  Errors:  {errors:,}")

    print("\nUpdated stats:")
    print_stats(conn)
    conn.close()

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NexusHub Meta Crawler")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max ghost nodes to crawl (default: all). Start with 50000")
    parser.add_argument("--stats", action="store_true",
                        help="Just show stats, don't crawl")
    parser.add_argument("--rebuild-queue", action="store_true",
                        help="Rebuild queue from scratch")
    parser.add_argument("--db", type=str, default=DB_PATH)
    args = parser.parse_args()

    if args.stats:
        conn = connect(args.db)
        ensure_schema(conn)
        print_stats(conn)
        conn.close()
        return

    if args.rebuild_queue:
        conn = connect(args.db)
        conn.execute("DELETE FROM meta_crawl_queue")
        conn.commit()
        conn.close()
        print("Queue cleared.")

    asyncio.run(run(args.db, args.limit))

if __name__ == "__main__":
    main()