"""
graph_analysis.py — NexusHub Graph Analysis (Rewritten)
---------------------------------------------------------
Three graphs built only from data you actually have:

  Graph 1 — Citation Graph (filtered)
    Only real crawled papers. No ghost nodes.
    Runs: PageRank, HITS, Louvain communities, Node2Vec

  Graph 2 — Concept Graph
    Nodes = concepts. Edges = co-occurrence in same paper.
    Runs: Concept PageRank, concept communities

  Graph 3 — Bridge Detection
    Which concepts span multiple fields (your core idea)
    Field overlap matrix

  Plus: Concept evolution over time

Usage:
    python graph_analysis.py --all
    python graph_analysis.py --citation
    python graph_analysis.py --concepts
    python graph_analysis.py --bridges
    python graph_analysis.py --evolution
    python graph_analysis.py --stats
"""

import sqlite3
import argparse
import json
import math
import collections
import numpy as np
import networkx as nx
from itertools import combinations

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("Warning: python-louvain not installed. pip install python-louvain")

try:
    from node2vec import Node2Vec
    HAS_NODE2VEC = True
except ImportError:
    HAS_NODE2VEC = False

DB_PATH = "knowledge.db"

# ── DB ────────────────────────────────────────────────────────────────────────

def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS paper_scores (
            paper_id        TEXT PRIMARY KEY,
            pagerank        REAL DEFAULT 0,
            community       INTEGER DEFAULT -1,
            hub_score       REAL DEFAULT 0,
            authority_score REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS concept_scores (
            concept_id   TEXT PRIMARY KEY,
            name         TEXT,
            pagerank     REAL DEFAULT 0,
            field_count  INTEGER DEFAULT 0,
            paper_count  INTEGER DEFAULT 0,
            bridge_score REAL DEFAULT 0,
            is_bridge    INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS concept_cooccurrence (
            concept_a TEXT,
            concept_b TEXT,
            weight    INTEGER DEFAULT 1,
            PRIMARY KEY (concept_a, concept_b)
        );
        CREATE TABLE IF NOT EXISTS field_overlap (
            field_a         TEXT,
            field_b         TEXT,
            shared_concepts INTEGER,
            overlap_score   REAL,
            PRIMARY KEY (field_a, field_b)
        );
        CREATE TABLE IF NOT EXISTS communities (
            community_id INTEGER PRIMARY KEY,
            label        TEXT,
            top_concepts TEXT,
            paper_count  INTEGER,
            avg_year     REAL
        );
        CREATE TABLE IF NOT EXISTS concept_evolution (
            concept_id    TEXT,
            year          INTEGER,
            paper_count   INTEGER,
            avg_citations REAL,
            PRIMARY KEY (concept_id, year)
        );
        CREATE TABLE IF NOT EXISTS node_embeddings (
            paper_id  TEXT PRIMARY KEY,
            embedding TEXT
        );
        CREATE TABLE IF NOT EXISTS bridge_concepts (
            concept_id   TEXT PRIMARY KEY,
            name         TEXT,
            fields       TEXT,
            field_count  INTEGER,
            paper_count  INTEGER,
            bridge_score REAL
        );
    """)
    conn.commit()

# ── STATS ─────────────────────────────────────────────────────────────────────

def print_stats(conn):
    print("\nDATABASE STATS")
    print("-" * 50)
    total         = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    with_abstract = conn.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL").fetchone()[0]
    with_concepts = conn.execute("SELECT COUNT(DISTINCT paper_id) FROM paper_concepts").fetchone()[0]
    total_concepts= conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    total_cites   = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    real_edges    = conn.execute("""
        SELECT COUNT(*) FROM citations c
        WHERE EXISTS (SELECT 1 FROM papers WHERE paper_id = c.paper_id)
          AND EXISTS (SELECT 1 FROM papers WHERE paper_id = c.cited_paper_id)
    """).fetchone()[0]
    ghost_nodes   = conn.execute("""
        SELECT COUNT(DISTINCT cited_paper_id) FROM citations
        WHERE cited_paper_id NOT IN (SELECT paper_id FROM papers)
    """).fetchone()[0]
    print(f"  Total papers crawled:     {total:,}")
    print(f"  Papers with abstracts:    {with_abstract:,}")
    print(f"  Papers with concepts:     {with_concepts:,}")
    print(f"  Unique concepts:          {total_concepts:,}")
    print(f"  Total citation edges:     {total_cites:,}")
    print(f"  Real citation edges:      {real_edges:,}  (both papers crawled)")
    print(f"  Ghost nodes in citations: {ghost_nodes:,}  (referenced but not crawled)")
    if total > 0:
        print(f"  Data completeness:        {with_abstract/total*100:.1f}%")
    print()

# ── GRAPH 1: CITATION GRAPH (FILTERED) ───────────────────────────────────────

def build_citation_graph(conn):
    print("Building filtered citation graph (real papers only)...")
    rows = conn.execute("""
        SELECT paper_id, citation_count, year
        FROM papers WHERE abstract IS NOT NULL
    """).fetchall()

    real_papers = set()
    G = nx.DiGraph()
    for r in rows:
        G.add_node(r["paper_id"], citations=r["citation_count"] or 0, year=r["year"] or 0)
        real_papers.add(r["paper_id"])

    edges = conn.execute("SELECT paper_id, cited_paper_id FROM citations").fetchall()
    added = 0
    for e in edges:
        if e["paper_id"] in real_papers and e["cited_paper_id"] in real_papers:
            G.add_edge(e["paper_id"], e["cited_paper_id"])
            added += 1

    print(f"  Real nodes: {G.number_of_nodes():,}")
    print(f"  Real edges: {G.number_of_edges():,}")
    print(f"  Skipped ghost edges: {len(edges) - added:,}")
    return G, real_papers

def run_citation_analysis(conn):
    print("\n== CITATION GRAPH ANALYSIS ==")
    G, real_papers = build_citation_graph(conn)
    if G.number_of_nodes() == 0:
        print("  No papers with abstracts found.")
        return

    print("Computing PageRank...")
    pr = nx.pagerank(G, alpha=0.85, max_iter=300)

    print("Computing HITS...")
    try:
        hubs, authorities = nx.hits(G, max_iter=200, normalized=True)
    except Exception:
        hubs = {n: 0 for n in G.nodes()}
        authorities = {n: 0 for n in G.nodes()}

    for pid, score in pr.items():
        conn.execute("""
            INSERT INTO paper_scores (paper_id, pagerank, hub_score, authority_score)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                pagerank        = excluded.pagerank,
                hub_score       = excluded.hub_score,
                authority_score = excluded.authority_score
        """, (pid, score, hubs.get(pid, 0), authorities.get(pid, 0)))
    conn.commit()

    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 PageRank papers:")
    for pid, score in top:
        row = conn.execute("SELECT title, year FROM papers WHERE paper_id=?", (pid,)).fetchone()
        if row:
            print(f"  {score:.6f} | {row['year']} | {row['title']}")

    if HAS_LOUVAIN:
        print("\nDetecting communities (filtered graph)...")
        UG = G.to_undirected()
        isolates = list(nx.isolates(UG))
        UG.remove_nodes_from(isolates)
        print(f"  Removed {len(isolates):,} isolates, running on {UG.number_of_nodes():,} nodes...")

        if UG.number_of_nodes() > 0:
            partition = community_louvain.best_partition(UG, random_state=42)
            modularity = community_louvain.modularity(partition, UG)
            print(f"  Modularity: {modularity:.4f}")
            print(f"  Communities: {len(set(partition.values()))}")

            for pid, comm_id in partition.items():
                conn.execute("""
                    INSERT INTO paper_scores (paper_id, community)
                    VALUES (?, ?)
                    ON CONFLICT(paper_id) DO UPDATE SET community = excluded.community
                """, (pid, comm_id))
            conn.commit()
            _save_community_summaries(conn, partition)

    if HAS_NODE2VEC and G.number_of_nodes() > 10:
        print("\nRunning Node2Vec...")
        UG2 = G.to_undirected()
        UG2.remove_nodes_from(list(nx.isolates(UG2)))
        if UG2.number_of_nodes() > 0:
            n2v = Node2Vec(UG2, dimensions=128, walk_length=20,
                           num_walks=10, p=1, q=0.5, workers=4, quiet=True)
            model = n2v.fit(window=10, min_count=1, batch_words=4)
            saved = 0
            for pid in UG2.nodes():
                if pid in model.wv:
                    conn.execute("INSERT OR REPLACE INTO node_embeddings VALUES (?, ?)",
                                 (pid, json.dumps(model.wv[pid].tolist())))
                    saved += 1
            conn.commit()
            print(f"  Saved {saved:,} node embeddings")

def _save_community_summaries(conn, partition):
    comm_papers = collections.defaultdict(list)
    for pid, cid in partition.items():
        comm_papers[cid].append(pid)

    conn.execute("DELETE FROM communities")
    for cid, pids in comm_papers.items():
        placeholders = ",".join("?" * len(pids))
        concept_rows = conn.execute(f"""
            SELECT c.name, COUNT(*) as cnt
            FROM paper_concepts pc
            JOIN concepts c ON c.concept_id = pc.concept_id
            WHERE pc.paper_id IN ({placeholders})
            GROUP BY c.name ORDER BY cnt DESC LIMIT 5
        """, pids).fetchall()

        top_concepts = [r["name"] for r in concept_rows]
        label = " / ".join(top_concepts[:2]) if top_concepts else f"Community {cid}"

        year_row = conn.execute(f"""
            SELECT AVG(year) FROM papers
            WHERE paper_id IN ({placeholders}) AND year IS NOT NULL
        """, pids).fetchone()

        conn.execute("INSERT OR REPLACE INTO communities VALUES (?, ?, ?, ?, ?)",
                     (cid, label, json.dumps(top_concepts), len(pids),
                      year_row[0] if year_row else None))
    conn.commit()
    print(f"  Saved {len(comm_papers)} community summaries")

# ── GRAPH 2: CONCEPT GRAPH ────────────────────────────────────────────────────

def build_concept_graph(conn, min_score=0.3, min_papers=3):
    print("\n== CONCEPT GRAPH ANALYSIS ==")
    print("Building concept co-occurrence graph...")

    rows = conn.execute("""
        SELECT pc.paper_id, pc.concept_id, c.name
        FROM paper_concepts pc
        JOIN concepts c ON c.concept_id = pc.concept_id
        JOIN papers p ON p.paper_id = pc.paper_id
        WHERE pc.score >= ?
        ORDER BY pc.paper_id
    """, (min_score,)).fetchall()

    paper_concepts = collections.defaultdict(list)
    for r in rows:
        paper_concepts[r["paper_id"]].append((r["concept_id"], r["name"]))

    print(f"  Papers with concepts: {len(paper_concepts):,}")

    cooccurrence = collections.defaultdict(int)
    concept_names = {}
    for pid, concepts in paper_concepts.items():
        for cid, name in concepts:
            concept_names[cid] = name
        for (a_id, _), (b_id, _) in combinations(concepts, 2):
            key = tuple(sorted([a_id, b_id]))
            cooccurrence[key] += 1

    print(f"  Concept pairs found: {len(cooccurrence):,}")

    G = nx.Graph()
    for (a, b), weight in cooccurrence.items():
        if weight >= min_papers:
            G.add_node(a, name=concept_names.get(a, a))
            G.add_node(b, name=concept_names.get(b, b))
            G.add_edge(a, b, weight=weight)

    print(f"  Concept nodes (min {min_papers} co-papers): {G.number_of_nodes():,}")
    print(f"  Concept edges: {G.number_of_edges():,}")

    conn.execute("DELETE FROM concept_cooccurrence")
    for (a, b), weight in cooccurrence.items():
        if weight >= min_papers:
            conn.execute("INSERT OR REPLACE INTO concept_cooccurrence VALUES (?, ?, ?)", (a, b, weight))
    conn.commit()

    return G, concept_names, paper_concepts

def run_concept_analysis(conn):
    G, concept_names, paper_concepts = build_concept_graph(conn)
    if G.number_of_nodes() == 0:
        print("  No concept data found.")
        return

    print("Computing concept PageRank...")
    pr = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=300)

    concept_paper_count = collections.defaultdict(int)
    for pid, concepts in paper_concepts.items():
        for cid, _ in concepts:
            concept_paper_count[cid] += 1

    conn.execute("DELETE FROM concept_scores")
    for cid, score in pr.items():
        conn.execute("INSERT OR REPLACE INTO concept_scores VALUES (?, ?, ?, 0, ?, 0, 0)",
                     (cid, concept_names.get(cid, cid), score, concept_paper_count.get(cid, 0)))
    conn.commit()

    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:15]
    print("\nMost central concepts in knowledge graph:")
    for cid, score in top:
        print(f"  {score:.6f} | {concept_paper_count.get(cid,0):5d} papers | {concept_names.get(cid,cid)}")

    if HAS_LOUVAIN:
        print("\nDetecting concept communities...")
        partition = community_louvain.best_partition(G, weight="weight", random_state=42)
        modularity = community_louvain.modularity(partition, G, weight="weight")
        n_comm = len(set(partition.values()))
        print(f"  Concept communities: {n_comm}")
        print(f"  Modularity: {modularity:.4f}")

        comm_concepts = collections.defaultdict(list)
        for cid, comm_id in partition.items():
            comm_concepts[comm_id].append((cid, pr.get(cid, 0)))

        print("\nTop concept communities:")
        sorted_comms = sorted(comm_concepts.items(),
                              key=lambda x: sum(s for _, s in x[1]), reverse=True)[:10]
        for comm_id, concepts in sorted_comms:
            top_c = sorted(concepts, key=lambda x: x[1], reverse=True)[:4]
            names = [concept_names.get(cid, cid) for cid, _ in top_c]
            print(f"  [{comm_id}]: {' | '.join(names)}")

# ── GRAPH 3: BRIDGE DETECTION ─────────────────────────────────────────────────

def run_bridge_detection(conn):
    print("\n== BRIDGE CONCEPT DETECTION ==")
    print("Finding concepts that appear across multiple fields...")

    rows = conn.execute("""
        SELECT
            pc.concept_id,
            c.name as concept_name,
            field_c.name as field_name,
            COUNT(DISTINCT pc.paper_id) as paper_count
        FROM paper_concepts pc
        JOIN concepts c ON c.concept_id = pc.concept_id
        JOIN paper_concepts field_pc ON field_pc.paper_id = pc.paper_id
        JOIN concepts field_c ON field_c.concept_id = field_pc.concept_id
        JOIN papers p ON p.paper_id = pc.paper_id
        WHERE pc.score >= 0.3
          AND field_c.level <= 1
          AND field_c.concept_id != pc.concept_id
        GROUP BY pc.concept_id, field_c.name
        HAVING paper_count >= 2
    """).fetchall()

    concept_fields = collections.defaultdict(lambda: collections.defaultdict(int))
    concept_names = {}
    for r in rows:
        concept_fields[r["concept_id"]][r["field_name"]] += r["paper_count"]
        concept_names[r["concept_id"]] = r["concept_name"]

    bridges = []
    for cid, fields in concept_fields.items():
        field_count  = len(fields)
        total_papers = sum(fields.values())
        if field_count < 2:
            continue
        bridge_score = field_count * math.log(total_papers + 1)
        bridges.append({
            "concept_id":   cid,
            "name":         concept_names.get(cid, cid),
            "fields":       dict(fields),
            "field_count":  field_count,
            "paper_count":  total_papers,
            "bridge_score": bridge_score,
        })

    bridges.sort(key=lambda x: x["bridge_score"], reverse=True)

    conn.execute("DELETE FROM bridge_concepts")
    for b in bridges:
        conn.execute("INSERT OR REPLACE INTO bridge_concepts VALUES (?, ?, ?, ?, ?, ?)",
                     (b["concept_id"], b["name"], json.dumps(list(b["fields"].keys())),
                      b["field_count"], b["paper_count"], b["bridge_score"]))
        conn.execute("""
            UPDATE concept_scores SET
                field_count  = ?,
                bridge_score = ?,
                is_bridge    = ?
            WHERE concept_id = ?
        """, (b["field_count"], b["bridge_score"],
              1 if b["field_count"] >= 3 else 0, b["concept_id"]))
    conn.commit()

    print(f"\nTop cross-domain bridge concepts:")
    for b in bridges[:15]:
        fields_preview = list(b["fields"].keys())[:4]
        print(f"  [{b['field_count']} fields | {b['paper_count']} papers] {b['name']}")
        print(f"    -> {' | '.join(fields_preview)}")

    _compute_field_overlap(conn)

def _compute_field_overlap(conn):
    # Step 1: find which papers belong to each top-level field
    # Step 2: find all concepts in those papers
    # Step 3: compare concept sets between fields

    print("Computing field overlap...")

    # Get papers per top-level field
    field_papers = conn.execute("""
        SELECT field_c.name as field_name, pc.paper_id
        FROM paper_concepts pc
        JOIN concepts field_c ON field_c.concept_id = pc.concept_id
        WHERE field_c.level = 0
    """).fetchall()

    from collections import defaultdict
    field_paper_set = defaultdict(set)
    for r in field_papers:
        field_paper_set[r["field_name"]].add(r["paper_id"])

    # Get all concepts per paper
    paper_concept_set = defaultdict(set)
    rows = conn.execute("SELECT paper_id, concept_id FROM paper_concepts").fetchall()
    for r in rows:
        paper_concept_set[r["paper_id"]].add(r["concept_id"])

    # Build concept set per field
    field_concepts = defaultdict(set)
    for field, papers in field_paper_set.items():
        for pid in papers:
            field_concepts[field].update(paper_concept_set[pid])

    print(f"  Fields: {len(field_concepts)}")
    for f, concepts in list(field_concepts.items())[:3]:
        print(f"  {f}: {len(concepts)} concepts")

    # Compute pairwise Jaccard overlap
    fields = list(field_concepts.keys())
    conn.execute("DELETE FROM field_overlap")
    overlaps = []

    for i in range(len(fields)):
        for j in range(i+1, len(fields)):
            fa, fb = fields[i], fields[j]
            shared = len(field_concepts[fa] & field_concepts[fb])
            if shared == 0:
                continue
            union = len(field_concepts[fa] | field_concepts[fb])
            score = shared / union if union > 0 else 0
            overlaps.append((fa, fb, shared, score))

    overlaps.sort(key=lambda x: x[3], reverse=True)
    for fa, fb, shared, score in overlaps:
        conn.execute(
            "INSERT OR REPLACE INTO field_overlap VALUES (?, ?, ?, ?)",
            (fa, fb, shared, score)
        )
    conn.commit()

    print(f"\n  Field pairs computed: {len(overlaps)}")
    print("\nMost conceptually similar fields:")
    for fa, fb, shared, score in overlaps[:10]:
        print(f"  {score:.3f} | {shared} shared concepts | {fa}  <->  {fb}")
# ── CONCEPT EVOLUTION ─────────────────────────────────────────────────────────

def run_evolution(conn):
    print("\n== CONCEPT EVOLUTION ==")
    rows = conn.execute("""
        SELECT pc.concept_id, p.year, COUNT(*) as paper_count, AVG(p.citation_count) as avg_citations
        FROM paper_concepts pc
        JOIN papers p ON p.paper_id = pc.paper_id
        WHERE p.year IS NOT NULL AND p.year >= 1950
          AND pc.score > 0.3
        GROUP BY pc.concept_id, p.year
        ORDER BY pc.concept_id, p.year
    """).fetchall()

    conn.execute("DELETE FROM concept_evolution")
    for r in rows:
        conn.execute("INSERT OR REPLACE INTO concept_evolution VALUES (?, ?, ?, ?)",
                     (r["concept_id"], r["year"], r["paper_count"], r["avg_citations"]))
    conn.commit()
    print(f"  Saved {len(rows):,} concept-year data points")

    growing = conn.execute("""
        SELECT c.name, SUM(ce.paper_count) as total
        FROM concept_evolution ce
        JOIN concepts c ON c.concept_id = ce.concept_id
        WHERE ce.year >= (SELECT MAX(year) - 5 FROM concept_evolution)
        GROUP BY ce.concept_id
        ORDER BY total DESC LIMIT 15
    """).fetchall()

    print("\nFastest growing concepts (last 5 years):")
    for r in growing:
        print(f"  {r['total']:5d} papers | {r['name']}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NexusHub Graph Analysis")
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--citation",  action="store_true")
    parser.add_argument("--concepts",  action="store_true")
    parser.add_argument("--bridges",   action="store_true")
    parser.add_argument("--evolution", action="store_true")
    parser.add_argument("--stats",     action="store_true")
    parser.add_argument("--db",        type=str, default=DB_PATH)
    args = parser.parse_args()

    conn = connect(args.db)
    ensure_tables(conn)
    print_stats(conn)

    if args.stats:
        conn.close()
        return

    run_all = args.all or not any([args.citation, args.concepts, args.bridges, args.evolution])

    if run_all or args.citation:  run_citation_analysis(conn)
    if run_all or args.concepts:  run_concept_analysis(conn)
    if run_all or args.bridges:   run_bridge_detection(conn)
    if run_all or args.evolution: run_evolution(conn)

    conn.close()
    print("\nGraph analysis complete.")

if __name__ == "__main__":
    main()