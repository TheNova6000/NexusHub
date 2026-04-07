"""
embeddings.py — Vector Embedding Pipeline for NexusHub
-------------------------------------------------------
Embeds paper abstracts + titles using sentence-transformers
and stores them in SQLite using sqlite-vec.

Install deps:
    pip install sentence-transformers sqlite-vec numpy

Usage:
    python embeddings.py                  # embed all papers
    python embeddings.py --search "query" # semantic search
"""

import sqlite3
import sqlite_vec
import numpy as np
import argparse
import struct
import time
from sentence_transformers import SentenceTransformer

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DB_PATH        = "nexushub.db"
MODEL_NAME     = "all-MiniLM-L6-v2"   # fast, 384-dim, great for abstracts
EMBEDDING_DIM  = 384
BATCH_SIZE     = 64

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def serialize(vec: np.ndarray) -> bytes:
    """Convert numpy float32 array → raw bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)

def deserialize(blob: bytes) -> np.ndarray:
    """Convert raw bytes → numpy float32 array."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# ─── DB SETUP ─────────────────────────────────────────────────────────────────

def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)                   # load the vec extension
    conn.enable_load_extension(False)
    return conn

def ensure_tables(conn: sqlite3.Connection):
    conn.executescript(f"""
        -- Store raw embedding blobs
        CREATE TABLE IF NOT EXISTS paper_embeddings (
            paper_id   TEXT PRIMARY KEY,
            embedding  BLOB NOT NULL,
            model      TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- sqlite-vec virtual table for fast ANN search
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_papers
        USING vec0(
            paper_id    TEXT PRIMARY KEY,
            embedding   float[{EMBEDDING_DIM}]
        );
    """)
    conn.commit()

# ─── EMBEDDING ────────────────────────────────────────────────────────────────

def build_text(title: str | None, abstract: str | None) -> str:
    """Combine title + abstract into a single string to embed."""
    parts = []
    if title:
        parts.append(title.strip())
    if abstract:
        parts.append(abstract.strip())
    return " [SEP] ".join(parts) if parts else ""

def embed_all_papers(conn: sqlite3.Connection, model: SentenceTransformer):
    """Embed every paper that doesn't yet have an embedding."""

    cur = conn.cursor()

    cur.execute("""
        SELECT p.paper_id, p.title, p.abstract
        FROM papers p
        LEFT JOIN paper_embeddings pe ON p.paper_id = pe.paper_id
        WHERE pe.paper_id IS NULL
    """)
    rows = cur.fetchall()

    if not rows:
        print("✓ All papers already embedded.")
        return

    print(f"Embedding {len(rows)} papers with '{MODEL_NAME}'...")

    total   = 0
    batches = [rows[i:i+BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        ids   = [r[0] for r in batch]
        texts = [build_text(r[1], r[2]) for r in batch]

        # Skip empty texts
        valid = [(i, t) for i, t in zip(ids, texts) if t.strip()]
        if not valid:
            continue

        valid_ids, valid_texts = zip(*valid)

        vecs = model.encode(
            list(valid_texts),
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,   # unit vectors → cosine = dot product
        ).astype(np.float32)

        for pid, vec in zip(valid_ids, vecs):
            blob = serialize(vec)

            conn.execute(
                "INSERT OR REPLACE INTO paper_embeddings VALUES (?, ?, ?, datetime('now'))",
                (pid, blob, MODEL_NAME)
            )
            conn.execute(
                "INSERT OR REPLACE INTO vec_papers (paper_id, embedding) VALUES (?, ?)",
                (pid, blob)
            )

        total += len(valid_ids)

        if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
            conn.commit()
            print(f"  [{total}/{len(rows)}] batches committed...")

    conn.commit()
    print(f"✓ Done. Embedded {total} papers.")

# ─── SEARCH ───────────────────────────────────────────────────────────────────

def semantic_search(
    conn: sqlite3.Connection,
    model: SentenceTransformer,
    query: str,
    top_k: int = 10
) -> list[dict]:
    """
    Search papers by semantic similarity to a query string.
    Returns list of dicts with paper info + similarity score.
    """

    q_vec = model.encode(query, normalize_embeddings=True).astype(np.float32)
    q_blob = serialize(q_vec)

    # sqlite-vec KNN query
    rows = conn.execute(f"""
        SELECT
            v.paper_id,
            v.distance,
            p.title,
            p.year,
            p.journal,
            p.citation_count,
            p.abstract
        FROM vec_papers v
        JOIN papers p ON p.paper_id = v.paper_id
        WHERE v.embedding MATCH ?
          AND k = ?
        ORDER BY v.distance
    """, (q_blob, top_k)).fetchall()

    results = []
    for row in rows:
        pid, dist, title, year, journal, cites, abstract = row
        # sqlite-vec returns L2 distance; since vecs are normalized,
        # cosine_sim = 1 - (dist^2 / 2)
        sim = max(0.0, 1.0 - (dist ** 2) / 2.0)
        results.append({
            "paper_id":      pid,
            "title":         title,
            "year":          year,
            "journal":       journal,
            "citation_count": cites,
            "similarity":    round(sim, 4),
            "abstract":      (abstract or "")[:300] + "..." if abstract else None,
        })

    return results

# ─── CROSS-DOMAIN BRIDGE FINDER ───────────────────────────────────────────────

def find_cross_domain_bridges(
    conn: sqlite3.Connection,
    paper_id: str,
    top_k: int = 5
) -> list[dict]:
    """
    Given a paper, find semantically similar papers from DIFFERENT concept clusters.
    This is the core of cross-domain knowledge transfer.
    """

    # Get source paper's top concept
    row = conn.execute("""
        SELECT c.name
        FROM paper_concepts pc
        JOIN concepts c ON c.concept_id = pc.concept_id
        WHERE pc.paper_id = ?
        ORDER BY pc.score DESC
        LIMIT 1
    """, (paper_id,)).fetchone()

    source_concept = row[0] if row else None

    # Get source embedding
    row = conn.execute(
        "SELECT embedding FROM paper_embeddings WHERE paper_id = ?",
        (paper_id,)
    ).fetchone()

    if not row:
        return []

    source_vec = deserialize(row[0])
    source_blob = serialize(source_vec)

    # Find similar papers, then filter to different top concept
    candidates = conn.execute(f"""
        SELECT
            v.paper_id,
            v.distance,
            p.title,
            p.year,
            p.citation_count
        FROM vec_papers v
        JOIN papers p ON p.paper_id = v.paper_id
        WHERE v.embedding MATCH ?
          AND k = 50
          AND v.paper_id != ?
        ORDER BY v.distance
    """, (source_blob, paper_id)).fetchall()

    bridges = []
    for pid, dist, title, year, cites in candidates:

        # Get top concept of candidate
        c_row = conn.execute("""
            SELECT c.name
            FROM paper_concepts pc
            JOIN concepts c ON c.concept_id = pc.concept_id
            WHERE pc.paper_id = ?
            ORDER BY pc.score DESC
            LIMIT 1
        """, (pid,)).fetchone()

        candidate_concept = c_row[0] if c_row else None

        # Only keep if concept is different (cross-domain!)
        if candidate_concept and candidate_concept != source_concept:
            sim = max(0.0, 1.0 - (dist ** 2) / 2.0)
            bridges.append({
                "paper_id":       pid,
                "title":          title,
                "year":           year,
                "citation_count": cites,
                "source_field":   source_concept,
                "target_field":   candidate_concept,
                "similarity":     round(sim, 4),
            })

        if len(bridges) >= top_k:
            break

    return bridges

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NexusHub Embedding Pipeline")
    parser.add_argument("--search",  type=str, help="Semantic search query")
    parser.add_argument("--bridge",  type=str, help="Paper ID to find cross-domain bridges for")
    parser.add_argument("--top-k",   type=int, default=10)
    parser.add_argument("--db",      type=str, default=DB_PATH)
    args = parser.parse_args()

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    conn = connect(args.db)
    ensure_tables(conn)

    if args.search:
        # Just search, don't re-embed
        results = semantic_search(conn, model, args.search, top_k=args.top_k)
        print(f"\n🔍 Results for: '{args.search}'\n{'─'*60}")
        for i, r in enumerate(results, 1):
            print(f"{i:2}. [{r['similarity']:.3f}] {r['title']}")
            print(f"     {r['year']} | {r['journal']} | {r['citation_count']} citations")
            if r['abstract']:
                print(f"     {r['abstract'][:120]}...")
            print()

    elif args.bridge:
        results = find_cross_domain_bridges(conn, args.bridge, top_k=args.top_k)
        print(f"\n🌉 Cross-domain bridges for {args.bridge}\n{'─'*60}")
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r['title']}")
            print(f"   {r['source_field']}  →  {r['target_field']}")
            print()

    else:
        # Default: embed all unprocessed papers
        embed_all_papers(conn, model)

    conn.close()

if __name__ == "__main__":
    main()
