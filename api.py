"""
NexusHub Backend — FastAPI
Run: uvicorn api:app --reload --port 8000
"""

import json
import logging
import os
import struct
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import sqlite3

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ─── CONFIG ───────────────────────────────────────────────────────────────────

NEXUS_DB     = os.getenv("NEXUS_DB",     "nexushub.db")
KNOWLEDGE_DB = os.getenv("KNOWLEDGE_DB", "nexushub.db")
OPENALEX_BASE  = "https://api.openalex.org"
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "nexushub@example.com")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("nexushub")

# ─── DEBUG ─────────────────────────────

DEBUG_BRIDGES = True

def debug(msg):
    if DEBUG_BRIDGES:
        print("[BRIDGE DEBUG]", msg)

# ─── SCHEMA CACHE — populated at startup ──────────────────────────────────────
# Maps table_name → set of column names actually present in the DB
_COLS: dict[str, set[str]] = {}

def _has(table: str, col: str) -> bool:
    return col in _COLS.get(table, set())

def _safe(table: str, col: str, alias: str = "") -> str:
    """Return 'table.col [AS alias]' if column exists, else 'NULL [AS alias]'."""
    label = f" AS {alias}" if alias else (f" AS {col}" if not _has(table, col) else "")
    if _has(table, col):
        return f"{table}.{col}{' AS ' + alias if alias else ''}"
    return f"NULL{' AS ' + (alias or col)}"

# ─── DB HELPERS ───────────────────────────────────────────────────────────────

def get_nexus() -> sqlite3.Connection:
    conn = sqlite3.connect(NEXUS_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def get_knowledge() -> sqlite3.Connection:
    conn = sqlite3.connect(KNOWLEDGE_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as e:
        log.warning("sqlite-vec not loaded: %s — vector search disabled", e)
    return conn

# ─── LIFESPAN ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("NexusHub starting — DB: %s / %s", NEXUS_DB, KNOWLEDGE_DB)
    _load_schema()
    _ensure_schema()
    yield
    log.info("NexusHub shutting down")

def _load_schema():
    """Discover every column in nexushub.db and cache them."""
    global _COLS
    try:
        conn = get_nexus()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        for (tname,) in tables:
            try:
                cols = conn.execute(f"PRAGMA table_info({tname})").fetchall()
                _COLS[tname] = {c[1] for c in cols}
            except Exception:
                pass
        conn.close()
        log.info("Schema loaded — tables: %s", list(_COLS.keys()))
        # Log the papers columns so we can see what's available
        log.info("papers columns: %s", sorted(_COLS.get("papers", set())))
    except Exception as e:
        log.error("Schema load failed: %s", e)

def _ensure_schema():
    """Create tables the app writes to if missing."""
    with get_nexus() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS enriched_cache (
                paper_id    TEXT PRIMARY KEY,
                fetched_at  INTEGER NOT NULL,
                success     INTEGER NOT NULL DEFAULT 0,
                oa_data     TEXT
            );
        """)
        conn.commit()
    # Reload after potential creation
    _load_schema()

# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="NexusHub", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── EMBEDDING HELPER ─────────────────────────────────────────────────────────

def _embed_query(text: str) -> Optional[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        if not hasattr(_embed_query, "_model"):
            _embed_query._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vec = _embed_query._model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        log.debug("Embedding unavailable: %s", e)
        return None

def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)

# ─── SAFE COLUMN SELECTORS ────────────────────────────────────────────────────

def _papers_select() -> str:
    """Build a safe SELECT for the papers table using only existing columns."""
    always = ["paper_id", "title", "year", "journal", "citation_count", "abstract"]
    optional = ["doi", "community"]
    cols = [f"p.{c}" for c in always if _has("papers", c)]
    for c in optional:
        cols.append(f"p.{c}" if _has("papers", c) else f"NULL AS {c}")
    return ", ".join(cols)

def _scores_select() -> str:
    """Safe select for paper_scores columns."""
    out = []
    for c in ["pagerank", "hub_score", "authority_score"]:
        out.append(f"ps.{c}" if _has("paper_scores", c) else f"NULL AS {c}")
    return ", ".join(out)

def _has_scores() -> bool:
    return bool(_COLS.get("paper_scores"))

def _scores_join() -> str:
    if _has_scores():
        return "LEFT JOIN paper_scores ps ON ps.paper_id = p.paper_id"
    return ""

# ─── ENDPOINT: /search ────────────────────────────────────────────────────────


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=200),
):

    vector_results = []
    text_results = []

    # semantic search
    vec = _embed_query(q)
    if vec is not None:
        try:
            vector_results = _vector_search(vec, limit)
        except Exception as e:
            log.warning("Vector search failed: %s", e)

    # keyword search (authors / concepts / titles)
    text_results = _text_search(q, limit)

    # merge results
    seen = set()
    results = []

    for r in text_results + vector_results:
        pid = r["paper_id"]
        if pid not in seen:
            seen.add(pid)
            results.append(r)

    return {
        "query": q,
        "count": len(results[:limit]),
        "results": results[:limit]
    }


def _vector_search(vec: list[float], limit: int) -> list[dict]:
    blob = _vec_to_blob(vec)
    kconn = get_knowledge()
    rows = kconn.execute(
        "SELECT paper_id, vec_distance_cosine(embedding, ?) AS distance "
        "FROM vec_papers ORDER BY distance ASC LIMIT ?",
        (blob, limit),
    ).fetchall()
    kconn.close()

    if not rows:
        return []

    paper_ids = [r["paper_id"] for r in rows]
    dist_map  = {r["paper_id"]: float(r["distance"]) for r in rows}
    placeholders = ",".join("?" * len(paper_ids))

    nconn = get_nexus()
    sel = _papers_select()
    pr  = "ps.pagerank" if _has("paper_scores", "pagerank") else "NULL AS pagerank"
    sj  = _scores_join()
    papers = nconn.execute(
        f"SELECT {sel}, {pr} FROM papers p {sj} WHERE p.paper_id IN ({placeholders})",
        paper_ids,
    ).fetchall()
    nconn.close()

    result_map = {r["paper_id"]: dict(r) for r in papers}
    out = []
    for pid in paper_ids:
        if pid not in result_map:
            continue
        p = result_map[pid]
        p["similarity"] = max(0.0, 1.0 - dist_map[pid])
        out.append(p)
    return out


def _text_search(q: str, limit: int) -> list[dict]:

    nconn = get_nexus()

    like = f"%{q}%"

    rows = nconn.execute("""
        SELECT DISTINCT
            p.paper_id,
            p.title,
            p.year,
            p.journal,
            p.citation_count,
            p.abstract,
            0.5 AS similarity
        FROM papers p

        LEFT JOIN paper_authors pa
            ON pa.paper_id = p.paper_id

        LEFT JOIN authors a
            ON a.author_id = pa.author_id

        LEFT JOIN paper_concepts pc
            ON pc.paper_id = p.paper_id

        LEFT JOIN concepts c
            ON c.concept_id = pc.concept_id

        WHERE
            p.title LIKE ?
            OR p.abstract LIKE ?
            OR a.name LIKE ?
            OR c.name LIKE ?
            OR p.journal LIKE ?

        ORDER BY p.citation_count DESC
        LIMIT ?
    """, (like, like, like, like, like, limit)).fetchall()

    nconn.close()

    return [dict(r) for r in rows]


# ─── ENDPOINT: /paper/{paper_id} ─────────────────────────────────────────────

@app.get("/paper/{paper_id}")
async def get_paper(paper_id: str, enrich: bool = True):
    nconn = get_nexus()
    sel = _papers_select()
    sc  = _scores_select()
    sj  = _scores_join()

    row = nconn.execute(
        f"SELECT {sel}, {sc} FROM papers p {sj} WHERE p.paper_id = ?",
        (paper_id,),
    ).fetchone()

    # Authors
    authors = []
    if _COLS.get("authors") and _COLS.get("paper_authors"):
        try:
            author_rows = nconn.execute(
                """SELECT a.name FROM authors a
                   JOIN paper_authors pa ON pa.author_id = a.author_id
                   WHERE pa.paper_id = ? ORDER BY pa.position""",
                (paper_id,),
            ).fetchall()
            authors = [{"name": r["name"]} for r in author_rows]
        except Exception as e:
            log.debug("Authors query failed: %s", e)

    # Concepts
    concepts = []
    if _COLS.get("concepts") and _COLS.get("paper_concepts"):
        try:
            score_col = "pc.score" if _has("paper_concepts", "score") else "NULL AS score"
            concept_rows = nconn.execute(
                f"""SELECT c.name, {score_col} FROM concepts c
                    JOIN paper_concepts pc ON pc.concept_id = c.concept_id
                    WHERE pc.paper_id = ? ORDER BY pc.score DESC LIMIT 20""",
                (paper_id,),
            ).fetchall()
            concepts = [{"name": r["name"], "score": r["score"]} for r in concept_rows]
        except Exception as e:
            log.debug("Concepts query failed: %s", e)

    nconn.close()

    if not row:
        paper = {
            "paper_id": paper_id, "title": None, "year": None, "journal": None,
            "doi": None, "citation_count": None, "abstract": None,
            "authors": [], "concepts": [], "pagerank": None, "community": None,
            "hub_score": None, "authority_score": None,
            "enriched": False, "enrich_source": None,
        }
    else:
        paper = dict(row)
        paper["authors"]  = authors
        paper["concepts"] = concepts
        paper.setdefault("enriched", False)
        paper.setdefault("enrich_source", None)
        paper.setdefault("community", None)
        paper.setdefault("doi", None)
        paper.setdefault("hub_score", None)
        paper.setdefault("authority_score", None)

    # Auto-enrich if abstract missing
    if enrich and not paper.get("abstract"):
        enriched = await _enrich_paper(paper_id)
        if enriched.get("success"):
            data = enriched.get("data") or {}
            paper["abstract"]      = paper.get("abstract") or data.get("abstract")
            paper["authors"]       = paper["authors"] or _parse_oa_authors(data)
            paper["concepts"]      = paper["concepts"] or _parse_oa_concepts(data)
            paper["enriched"]      = True
            paper["enrich_source"] = enriched.get("source")
            if not paper.get("title"):  paper["title"]  = data.get("title")
            if not paper.get("year"):   paper["year"]   = data.get("year")
            if not paper.get("doi"):    paper["doi"]    = data.get("doi")

    return paper

# ─── ENDPOINT: /paper/{paper_id}/graph ───────────────────────────────────────

@app.get("/paper/{paper_id}/graph")
async def get_graph(
    paper_id: str,
    depth: int = Query(2, ge=1, le=3),
    limit: int = Query(50, ge=1, le=200),
):
    nconn = get_nexus()

    # Detect citation table name
    cit_table = None
    for candidate in ["citations", "paper_citations", "paper_references"]:
        if candidate in _COLS:
            cit_table = candidate
            break

    center_row = nconn.execute(
        f"""SELECT p.paper_id, p.title, p.year,
               {'p.citation_count' if _has('papers','citation_count') else 'NULL AS citation_count'},
               {'ps.pagerank' if _has_scores() else 'NULL AS pagerank'}
            FROM papers p {_scores_join()}
            WHERE p.paper_id = ?""",
        (paper_id,),
    ).fetchone()

    center = {
        "id": paper_id,
        "paper_id": paper_id,
        "title": center_row["title"] if center_row else paper_id,
        "year": center_row["year"] if center_row else None,
        "citation_count": (center_row["citation_count"] if center_row else None) or 0,
        "type": "center",
    }

    nodes: dict[str, dict] = {paper_id: center}
    edges: list[dict] = []
    visited = {paper_id}
    frontier = [paper_id]

    if cit_table:
        # Discover column names in citations table
        cit_cols = _COLS.get(cit_table, set())
        # Determine which column is "source" and which is "target"
        # Common schemas: (paper_id, cited_paper_id) or (citing_id, cited_id)
        src_col  = next((c for c in ["paper_id", "citing_id", "source"] if c in cit_cols), None)
        tgt_col  = next((c for c in ["cited_paper_id", "cited_id", "target"] if c in cit_cols), None)

        for _d in range(depth):
            if len(nodes) >= limit:
                break
            next_frontier = []
            for pid in frontier:
                if len(nodes) >= limit:
                    break
                batch = max(1, (limit - len(nodes)) // max(1, len(frontier)))

                # Papers that cite this one
                if src_col and tgt_col:
                    try:
                        citers = nconn.execute(
                            f"""SELECT c.{src_col} AS citer_id, p.title, p.year,
                                       {'p.citation_count' if _has('papers','citation_count') else 'NULL AS citation_count'}
                                FROM {cit_table} c
                                LEFT JOIN papers p ON p.paper_id = c.{src_col}
                                WHERE c.{tgt_col} = ? LIMIT ?""",
                            (pid, batch),
                        ).fetchall()
                        for r in citers:
                            cid = r["citer_id"]
                            if cid and cid not in nodes:
                                nodes[cid] = {"id": cid, "paper_id": cid,
                                              "title": r["title"] or cid,
                                              "year": r["year"],
                                              "citation_count": r["citation_count"] or 0,
                                              "type": "citing"}
                                next_frontier.append(cid)
                            if cid and not any(e["source"]==cid and e["target"]==pid for e in edges):
                                edges.append({"source": cid, "target": pid})
                    except Exception as e:
                        log.debug("Citing query failed: %s", e)

                    try:
                        refs = nconn.execute(
                            f"""SELECT c.{tgt_col} AS ref_id, p.title, p.year,
                                       {'p.citation_count' if _has('papers','citation_count') else 'NULL AS citation_count'}
                                FROM {cit_table} c
                                LEFT JOIN papers p ON p.paper_id = c.{tgt_col}
                                WHERE c.{src_col} = ? LIMIT ?""",
                            (pid, batch),
                        ).fetchall()
                        for r in refs:
                            rid = r["ref_id"]
                            if rid and rid not in nodes:
                                nodes[rid] = {"id": rid, "paper_id": rid,
                                              "title": r["title"] or rid,
                                              "year": r["year"],
                                              "citation_count": r["citation_count"] or 0,
                                              "type": "reference"}
                                next_frontier.append(rid)
                            if rid and not any(e["source"]==pid and e["target"]==rid for e in edges):
                                edges.append({"source": pid, "target": rid})
                    except Exception as e:
                        log.debug("Refs query failed: %s", e)

            frontier = [n for n in next_frontier if n not in visited]
            visited.update(frontier)

    nconn.close()
    node_list = list(nodes.values())
    return {"center": center, "nodes": node_list, "edges": edges,
            "node_count": len(node_list), "edge_count": len(edges)}

# ─── ENDPOINT: /paper/{paper_id}/bridges ─────────────────────────────────────


@app.get("/paper/{paper_id}/bridges")
async def get_bridges(paper_id: str, limit: int = Query(6, ge=1, le=20)):

    debug(f"Starting bridge detection for {paper_id}")

    nconn = get_nexus()

    # ───────── SOURCE CONCEPTS ─────────

    rows = nconn.execute("""
        SELECT c.name, c.level
        FROM paper_concepts pc
        JOIN concepts c ON c.concept_id = pc.concept_id
        WHERE pc.paper_id = ?
    """, (paper_id,)).fetchall()

    debug(f"Concept rows found: {len(rows)}")

    source_concepts = set()
    source_field = None

    for r in rows:
        source_concepts.add(r["name"])

        if r["level"] in (1,2) and source_field is None:
            source_field = r["name"]

    debug(f"Source concepts: {len(source_concepts)}")
    debug(f"Source field: {source_field}")

    if not source_concepts:
        debug("No concepts found — exiting")
        nconn.close()
        return {"paper_id": paper_id, "source_field": None, "bridges": []}

    # ───────── EMBEDDING ─────────

    kconn = get_knowledge()

    emb = kconn.execute(
        "SELECT embedding FROM paper_embeddings WHERE paper_id = ?",
        (paper_id,)
    ).fetchone()

    if not emb:
        debug("No embedding found!")
        kconn.close()
        nconn.close()
        return {"paper_id": paper_id, "source_field": source_field, "bridges": []}

    debug("Embedding found")

    blob = bytes(emb["embedding"])
    n_f = len(blob) // 4
    vec = struct.unpack(f"{n_f}f", blob)

    # ───────── VECTOR SEARCH ─────────

    debug("Running vector similarity search")

    neighbors = kconn.execute("""
        SELECT paper_id,
               vec_distance_cosine(embedding, ?) AS dist
        FROM vec_papers
        WHERE paper_id != ?
        ORDER BY dist ASC
        LIMIT 200
    """, (_vec_to_blob(list(vec)), paper_id)).fetchall()

    debug(f"Neighbors found: {len(neighbors)}")

    kconn.close()

    if not neighbors:
        debug("Vector search returned nothing")
        nconn.close()
        return {"paper_id": paper_id, "source_field": source_field, "bridges": []}

    candidate_ids = [r["paper_id"] for r in neighbors]
    dist_map = {r["paper_id"]: r["dist"] for r in neighbors}

    # ───────── LOAD CANDIDATE CONCEPTS ─────────

    placeholders = ",".join("?" * len(candidate_ids))

    rows = nconn.execute(f"""
        SELECT p.paper_id, p.title, p.year, c.name AS concept
        FROM papers p
        LEFT JOIN paper_concepts pc ON pc.paper_id = p.paper_id
        LEFT JOIN concepts c ON c.concept_id = pc.concept_id
        WHERE p.paper_id IN ({placeholders})
    """, candidate_ids).fetchall()

    debug(f"Candidate rows loaded: {len(rows)}")

    paper_concepts = {}
    paper_meta = {}

    for r in rows:

        pid = r["paper_id"]

        paper_meta.setdefault(pid,{
            "title": r["title"],
            "year": r["year"]
        })

        paper_concepts.setdefault(pid,set())

        if r["concept"]:
            paper_concepts[pid].add(r["concept"])

    debug(f"Papers grouped: {len(paper_concepts)}")

    # ───────── BRIDGE FILTER ─────────

    bridges = []

    for pid in paper_concepts:
        candidate_concepts = paper_concepts[pid]

        overlap = source_concepts & candidate_concepts

        similarity = max(0.0, 1.0 - dist_map.get(pid, 1.0))

        debug(f"{pid} similarity={similarity:.3f} overlap={len(overlap)}")

        # bridge = similar but NOT same domain
        if similarity < 0.55:
            continue

        if len(overlap) > 3:
            continue



        bridges.append({
            "paper_id": pid,
            "title": paper_meta[pid]["title"],
            "year": paper_meta[pid]["year"],
            "similarity": similarity,
            "shared_concepts": list(overlap)
        })
        debug("Top neighbors:")
        for n in neighbors[:10]:
            debug(f"{n['paper_id']} dist={n['dist']:.3f}")

        if len(bridges) >= limit:
            break

    debug(f"Bridges found: {len(bridges)}")

    nconn.close()

    return {
        "paper_id": paper_id,
        "source_field": source_field,
        "bridges": bridges
    }

# ─── ENDPOINT: /paper/{paper_id}/enrich ──────────────────────────────────────

@app.get("/paper/{paper_id}/enrich")
async def enrich_paper(paper_id: str, force: bool = False):
    return await _enrich_paper(paper_id, force=force)


async def _enrich_paper(paper_id: str, force: bool = False) -> dict:
    nconn = get_nexus()

    if not force and _has("papers", "abstract"):
        full_row = nconn.execute(
            "SELECT abstract FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if full_row and full_row["abstract"]:
            nconn.close()
            return {"paper_id": paper_id, "source": "local_db", "already_full": True,
                    "data": None, "success": True, "message": "Full data in local DB"}

    if not force:
        cache_row = nconn.execute(
            "SELECT fetched_at, success, oa_data FROM enriched_cache WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        if cache_row:
            nconn.close()
            data = json.loads(cache_row["oa_data"]) if cache_row["oa_data"] else None
            return {"paper_id": paper_id, "source": "cache", "already_full": False,
                    "data": data, "success": bool(cache_row["success"]),
                    "message": "Served from enriched_cache"}

    nconn.close()
    oa_data = await _fetch_openalex(paper_id)
    _store_cache(paper_id, oa_data)

    return {"paper_id": paper_id, "source": "openalex", "already_full": False,
            "data": oa_data, "success": oa_data is not None,
            "message": "Fetched from OpenAlex" if oa_data else "Not found in OpenAlex"}


async def _fetch_openalex(paper_id: str) -> Optional[dict]:
    headers = {"User-Agent": f"NexusHub/2.0 (mailto:{OPENALEX_EMAIL})"}
    pid = str(paper_id).strip()

    urls = []
    if pid.upper().startswith("W") and pid[1:].isdigit():
        urls.append(f"{OPENALEX_BASE}/works/{pid}")
    elif pid.startswith("10."):
        urls.append(f"{OPENALEX_BASE}/works/https://doi.org/{pid}")
    else:
        urls.append(f"{OPENALEX_BASE}/works/{pid}")

    async with httpx.AsyncClient(timeout=15) as client:
        for url in urls:
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    raw = resp.json()
                    if "results" in raw:
                        raw = (raw["results"] or [None])[0]
                    if raw:
                        return _parse_oa_work(raw)
            except Exception as e:
                log.warning("OpenAlex fetch error %s: %s", paper_id, e)
    return None


def _parse_oa_work(raw: dict) -> dict:
    abstract = None
    if raw.get("abstract_inverted_index"):
        abstract = _reconstruct_abstract(raw["abstract_inverted_index"])

    authors = []
    for a in raw.get("authorships", []):
        name = (a.get("author") or {}).get("display_name")
        if name:
            authors.append({"name": name})

    concepts = []
    for c in raw.get("concepts", []):
        name = c.get("display_name") or c.get("name")
        if name:
            concepts.append({"name": name, "score": c.get("score", 0)})

    doi = raw.get("doi", "") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    loc = raw.get("primary_location") or {}
    journal = (loc.get("source") or {}).get("display_name")

    return {
        "paper_id": raw.get("id", "").split("/")[-1],
        "title": raw.get("display_title") or raw.get("title"),
        "abstract": abstract,
        "year": raw.get("publication_year"),
        "doi": doi or None,
        "journal": journal,
        "citation_count": raw.get("cited_by_count"),
        "authors": authors,
        "concepts": concepts,
    }


def _parse_oa_authors(data: dict) -> list[dict]:
    return data.get("authors", [])

def _parse_oa_concepts(data: dict) -> list[dict]:
    return data.get("concepts", [])

def _reconstruct_abstract(inverted: dict) -> str:
    pos_word = {}
    for word, positions in inverted.items():
        for p in positions:
            pos_word[p] = word
    return " ".join(pos_word[k] for k in sorted(pos_word))

def _store_cache(paper_id: str, data: Optional[dict]):
    with get_nexus() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO enriched_cache (paper_id, fetched_at, success, oa_data) "
            "VALUES (?, ?, ?, ?)",
            (paper_id, int(time.time()), 1 if data else 0,
             json.dumps(data) if data else None),
        )
        conn.commit()

# ─── ENDPOINT: /communities ───────────────────────────────────────────────────

@app.get("/communities")
async def get_communities():
    if "communities" not in _COLS:
        return []
    nconn = get_nexus()
    cols = _COLS["communities"]
    sel  = ", ".join(
        c if c in cols else f"NULL AS {c}"
        for c in ["community_id", "label", "top_concepts", "paper_count", "avg_year"]
    )
    rows = nconn.execute(
        f"SELECT {sel} FROM communities ORDER BY {'paper_count DESC' if 'paper_count' in cols else 'community_id'}"
    ).fetchall()
    nconn.close()

    out = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("top_concepts"), str):
            try:    d["top_concepts"] = json.loads(d["top_concepts"])
            except: d["top_concepts"] = [d["top_concepts"]]
        out.append(d)
    return out

# ─── ENDPOINT: /community/{id}/papers ────────────────────────────────────────

@app.get("/community/{community_id}/papers")
async def get_community_papers(community_id: int, limit: int = Query(20, ge=1, le=100)):
    if not _has("papers", "community"):
        return []
    nconn = get_nexus()
    pr  = "ps.pagerank" if _has_scores() else "NULL AS pagerank"
    sj  = _scores_join()
    rows = nconn.execute(
        f"""SELECT p.paper_id, p.title, p.year,
                   {'p.citation_count' if _has('papers','citation_count') else 'NULL AS citation_count'},
                   {pr}
            FROM papers p {sj}
            WHERE p.community = ?
            ORDER BY {'ps.pagerank DESC' if _has_scores() else 'p.paper_id'}
            LIMIT ?""",
        (community_id, limit),
    ).fetchall()
    nconn.close()
    return [dict(r) for r in rows]

# ─── ENDPOINT: /trending ──────────────────────────────────────────────────────

@app.get("/trending")
async def get_trending(limit: int = Query(30, ge=1, le=100)):
    nconn = get_nexus()
    sel = _papers_select()
    pr  = "ps.pagerank" if _has_scores() else "NULL AS pagerank"
    sj  = _scores_join()

    # If paper_scores doesn't exist, fallback to citation_count ordering
    if _has_scores():
        where = "WHERE p.abstract IS NOT NULL" if _has("papers", "abstract") else ""
        rows = nconn.execute(
            f"SELECT {sel}, {pr} FROM papers p {sj} {where} "
            f"ORDER BY ps.pagerank DESC LIMIT ?",
            (limit,),
        ).fetchall()
    else:
        order = "p.citation_count DESC" if _has("papers", "citation_count") else "p.paper_id"
        where = "WHERE p.abstract IS NOT NULL" if _has("papers", "abstract") else ""
        rows = nconn.execute(
            f"SELECT {sel}, {pr} FROM papers p {where} ORDER BY {order} LIMIT ?",
            (limit,),
        ).fetchall()

    nconn.close()
    return [dict(r) for r in rows]

# ─── ENDPOINT: /fields ────────────────────────────────────────────────────────

@app.get("/fields")
async def get_fields():
    if "field_overlap" in _COLS:
        nconn = get_nexus()
        rows = nconn.execute(
            "SELECT field_a, field_b, shared_concepts, overlap_score "
            "FROM field_overlap ORDER BY overlap_score DESC LIMIT 40"
        ).fetchall()
        nconn.close()
        if rows:
            return [dict(r) for r in rows]

    return await _synthesise_fields()


async def _synthesise_fields() -> list[dict]:
    if not (_COLS.get("paper_concepts") and _COLS.get("concepts")):
        return []
    nconn = get_nexus()
    lvl = "AND a.level = 0 AND b.level = 0" if _has("concepts", "level") else ""
    try:
        rows = nconn.execute(
            f"""SELECT a.name AS field_a, b.name AS field_b,
                       COUNT(DISTINCT pa.paper_id) AS shared_concepts,
                       CAST(COUNT(DISTINCT pa.paper_id) AS REAL) /
                         MAX(1, (SELECT COUNT(*) FROM papers)) AS overlap_score
                FROM paper_concepts pa
                JOIN concepts a ON a.concept_id = pa.concept_id
                JOIN paper_concepts pb ON pb.paper_id = pa.paper_id
                JOIN concepts b ON b.concept_id = pb.concept_id
                              AND b.concept_id > a.concept_id
                WHERE 1=1 {lvl}
                GROUP BY a.name, b.name
                ORDER BY shared_concepts DESC LIMIT 20"""
        ).fetchall()
        nconn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        nconn.close()
        log.warning("Synthesise fields failed: %s", e)
        return []

# ─── ENDPOINT: /bridge-concepts ──────────────────────────────────────────────

@app.get("/bridge-concepts")
async def get_bridge_concepts(limit: int = Query(20, ge=1, le=50)):
    if "bridge_concepts" not in _COLS:
        return []
    nconn = get_nexus()
    bc_cols = _COLS["bridge_concepts"]
    sel = ", ".join(
        f"bc.{c}" if c in bc_cols else f"NULL AS {c}"
        for c in ["concept_id", "name", "fields", "field_count", "bridge_score"]
    )
    pc_join = "LEFT JOIN paper_concepts pc ON pc.concept_id = bc.concept_id" \
              if _COLS.get("paper_concepts") else ""
    pc_count = "COUNT(pc.paper_id) AS paper_count" if _COLS.get("paper_concepts") else "0 AS paper_count"
    order = "bc.bridge_score DESC" if "bridge_score" in bc_cols else "bc.name"

    rows = nconn.execute(
        f"SELECT {sel}, {pc_count} FROM bridge_concepts bc {pc_join} "
        f"GROUP BY bc.concept_id ORDER BY {order} LIMIT ?",
        (limit,),
    ).fetchall()
    nconn.close()

    out = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("fields"), str):
            try:    d["fields"] = json.loads(d["fields"])
            except: d["fields"] = [d["fields"]]
        out.append(d)
    return out

# ─── ENDPOINT: /evolution ─────────────────────────────────────────────────────

@app.get("/evolution")
async def get_evolution(concept: str = Query("neural networks"), top_n: int = Query(10, ge=1, le=20)):
    if "concept_evolution" not in _COLS or "concepts" not in _COLS:
        return {"concepts": {}}
    nconn = get_nexus()
    ac_col = "ce.avg_citations" if _has("concept_evolution", "avg_citations") else "NULL AS avg_citations"
    try:
        rows = nconn.execute(
            f"""SELECT ce.year, ce.paper_count, {ac_col}, c.name AS concept_name
                FROM concept_evolution ce
                JOIN concepts c ON c.concept_id = ce.concept_id
                WHERE LOWER(c.name) LIKE ?
                ORDER BY c.name, ce.year""",
            (f"%{concept.lower()}%",),
        ).fetchall()
    except Exception as e:
        nconn.close()
        log.warning("Evolution query failed: %s", e)
        return {"concepts": {}}

    nconn.close()
    result: dict[str, list] = {}
    for r in rows:
        cn = r["concept_name"]
        result.setdefault(cn, []).append({"year": r["year"], "paper_count": r["paper_count"]})

    sorted_c = sorted(
        result.items(), key=lambda x: sum(e["paper_count"] for e in x[1]), reverse=True
    )[:top_n]
    return {"concepts": dict(sorted_c)}

# ─── ENDPOINT: /cache/stats ──────────────────────────────────────────────────

@app.get("/cache/stats")
async def get_cache_stats():
    nconn = get_nexus()
    total = nconn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]

    with_abstract = 0
    if _has("papers", "abstract"):
        with_abstract = nconn.execute(
            "SELECT COUNT(*) AS n FROM papers WHERE abstract IS NOT NULL AND abstract != ''"
        ).fetchone()["n"]

    cache_total = cache_success = 0
    if "enriched_cache" in _COLS:
        cache_total   = nconn.execute("SELECT COUNT(*) AS n FROM enriched_cache").fetchone()["n"]
        cache_success = nconn.execute(
            "SELECT COUNT(*) AS n FROM enriched_cache WHERE success = 1"
        ).fetchone()["n"]

    nconn.close()
    return {
        "total_papers":         total,
        "papers_with_abstract": with_abstract,
        "meta_only_papers":     total - with_abstract,
        "cache_total":          cache_total,
        "cache_success":        cache_success,
        "cache_failed":         cache_total - cache_success,
        "enrichment_rate":      round(cache_success / total, 4) if total else 0,
    }

# ─── HEALTH ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        get_nexus().execute("SELECT 1")
        return {"status": "ok", "db": True, "tables": list(_COLS.keys())}
    except Exception as e:
        return {"status": "degraded", "db": False, "error": str(e)}