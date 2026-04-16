# NexusHub

**A citation-graph search engine for mapping the structure of human knowledge.**

NexusHub treats research literature not as a flat pile of documents, but as what it actually is — a directed graph where ideas build on other ideas. By identifying which papers sit at the structural center of that graph, NexusHub lets you navigate knowledge by its underlying architecture, not just by keywords.

---

## The Core Idea

Most academic search engines answer the question: *"Which papers mention this term?"*

NexusHub answers a different question: *"How does this idea connect to everything else?"*

Research knowledge is not uniformly distributed. A small number of papers are *cross-domain generative* — their ideas escaped their original field and seeded multiple others. Claude Shannon's 1948 paper on information theory didn't just start information theory. It seeded data compression, cryptography, statistical mechanics, neuroscience, and genetics. These papers are not just highly cited within a field. They are ontological pivots — they changed what questions were even askable.

NexusHub calls these **Nexus Papers**.

### The Knowledge Pyramid

All research knowledge can be thought of as a three-layer pyramid:

```
         ┌─────────────────────────────┐
         │      NEXUS PAPERS           │  ← Cross-domain generative works.
         │  (cross-domain anchors)     │    A few hundred papers across all
         │                             │    of science. Their web of mutual
         └─────────────────────────────┘    connections = the NexusHub.
       ┌──────────────────────────────────┐
       │      DOMAIN CORNERSTONES         │  ← Foundational within a field.
       │  (field-defining works)          │    Not cross-domain, but essential.
       └──────────────────────────────────┘
     ┌────────────────────────────────────────┐
     │         THE LONG TAIL                  │  ← Specialized, narrow-scope work.
     │  (specialized research)                │    Important locally, not globally.
     └────────────────────────────────────────┘
```

Starting from Nexus Papers and crawling outward through their citation network — both downward into what they built on and upward into what built on them — covers 60–80% of meaningful human knowledge from a surprisingly small seed. Knowledge is hierarchically structured and heavily networked. Most of it flows through a handful of structural anchors.

---

## How It Works

### 1. Bidirectional Citation Crawling (`meta_crawl.py`)

The crawler uses the [OpenAlex API](https://openalex.org) (250M+ academic works, fully open) to build the citation graph. It crawls in **both directions**:

- **Downward through references** — what foundational work does this paper build on?
- **Upward through citations** — what newer work built on top of this paper?

For each fully crawled paper, it stores: title, abstract, year, journal, citation count, authors, and OpenAlex concept tags.

For *ghost papers* (papers that appear in citation lists but haven't been fully crawled), `meta_crawl.py` runs a lightweight second pass — fetching only title, year, citation count, and top 3 concepts, without abstract or full author list. This converts dead-end nodes into real graph nodes with enough metadata to make the graph dense and analysable. Ghost node resolution is prioritised by citation frequency — the most-cited ghost nodes are resolved first.

```bash
python meta_crawl.py --limit 50000   # crawl top 50k ghost nodes by citation count
python meta_crawl.py --stats         # show current graph coverage
python meta_crawl.py --rebuild-queue # reset and rebuild the crawl queue
```

### 2. Graph Analysis (`graph_analysis.py`)

Once the citation network is built, NexusHub runs structural analysis across three graph types:

**Graph 1 — Citation Graph (ghost-filtered)**

Only papers with full records are included as nodes. Ghost edges are explicitly excluded to maintain graph integrity. Runs:

- **PageRank** (α=0.85) — identifies globally important papers by how many paths flow through them, not just raw citation count
- **HITS (Hubs & Authorities)** — hub score identifies papers that point to many important works; authority score identifies papers that are pointed to by many important papers
- **Louvain community detection** — finds densely connected clusters of papers, each labelled by their dominant concepts
- **Node2Vec** — generates 128-dimensional graph embeddings for each paper, capturing structural position in the citation network

**Graph 2 — Concept Co-occurrence Graph**

Nodes are concepts (from OpenAlex's concept taxonomy). Edges are weighted by how many papers share both concepts. Runs concept-level PageRank and community detection to find which concepts are most central to the overall knowledge graph.

**Graph 3 — Bridge Detection**

The most original analysis. A *bridge concept* is one that appears across papers from multiple distinct top-level fields. The bridge score is `field_count × log(total_papers + 1)`. These are the concepts that connect disciplines — the places where ideas jump fields.

Also computes a **field overlap matrix** using Jaccard similarity between each pair of fields' concept sets. Tells you which fields share the most conceptual vocabulary, which is a proxy for which fields are most likely to cross-pollinate.

```bash
python graph_analysis.py --all        # run all analyses
python graph_analysis.py --citation   # citation graph only
python graph_analysis.py --concepts   # concept graph only
python graph_analysis.py --bridges    # bridge detection only
python graph_analysis.py --evolution  # concept growth over time
python graph_analysis.py --stats      # show DB stats (ghost counts, coverage)
```

### 3. Semantic Search via Vector Embeddings (`embeddings.py`)

Every paper's abstract is encoded into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`. Stored via `sqlite-vec` for fast in-database cosine similarity search.

This means you can search by *meaning*, not keywords. "How does entropy apply to biological systems?" retrieves papers conceptually related to that question even if they use none of those exact words.

### 4. Search (`/search` endpoint)

Every search query runs two passes in parallel and merges results:

- **Semantic pass** — encodes the query into a vector and runs cosine similarity against the embedding store
- **Keyword pass** — SQL LIKE query across title, abstract, author names, concept names, and journal name

Results are deduplicated, ranked, and returned with similarity scores.

### 5. Bridge Paper Detection (`/paper/{id}/bridges`)

Given any paper, this endpoint finds papers in *different* fields that are nonetheless semantically similar to it. The definition of a bridge: cosine similarity ≥ 0.55 *and* fewer than 3 shared concepts with the source paper. High similarity but low concept overlap = the same idea expressed in a different domain's vocabulary.

### 6. FastAPI Backend + SQLite Storage

All data lives in two SQLite databases:
- `nexushub.db` — papers, authors, concepts, citations, communities, bridge concepts, field overlap, concept evolution
- Embedding store (via `sqlite-vec`) — paper vectors for semantic search

The API is schema-adaptive at startup — it reads the actual column set from the live database and builds all queries dynamically, so it degrades gracefully if optional columns or tables are missing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NexusHub System                             │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  meta_crawl  │    │   graph_     │    │    embeddings.py     │  │
│  │  .py         │    │   analysis   │    │                      │  │
│  │              │    │   .py        │    │  sentence-           │  │
│  │  OpenAlex    │───▶│              │    │  transformers        │  │
│  │  API crawl   │    │  PageRank    │    │  all-MiniLM-L6-v2    │  │
│  │  Bidirect.   │    │  HITS        │    │                      │  │
│  │  citations   │    │  Louvain     │    │  sqlite-vec          │  │
│  │  Ghost node  │    │  Node2Vec    │    │  vector store        │  │
│  │  resolution  │    │  Bridge      │    │                      │  │
│  └──────┬───────┘    │  detection   │    └──────────┬───────────┘  │
│         │            └──────┬───────┘               │              │
│         ▼                   ▼                        ▼              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    nexushub.db (SQLite)                      │   │
│  │  papers / authors / concepts / citations / paper_scores     │   │
│  │  communities / bridge_concepts / field_overlap              │   │
│  │  concept_evolution / concept_cooccurrence / node_embeddings │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                    ┌─────────────────┐                             │
│                    │    api.py       │                             │
│                    │    FastAPI      │                             │
│                    │                 │                             │
│                    │  /search        │                             │
│                    │  /paper/{id}    │                             │
│                    │  /paper/{id}/   │                             │
│                    │    graph        │                             │
│                    │  /paper/{id}/   │                             │
│                    │    bridges      │                             │
│                    │  /trending      │                             │
│                    │  /communities   │                             │
│                    │  /fields        │                             │
│                    │  /bridge-       │                             │
│                    │    concepts     │                             │
│                    │  /evolution     │                             │
│                    └────────┬────────┘                             │
│                             │                                      │
│                    ┌────────▼────────┐                             │
│                    │   index.html    │                             │
│                    │   Web UI        │                             │
│                    └─────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /search?q=...` | Hybrid semantic + keyword search. Returns merged, deduplicated results with similarity scores. |
| `GET /paper/{id}` | Full paper detail. Auto-enriches from OpenAlex if abstract is missing locally. Caches enrichment results. |
| `GET /paper/{id}/graph?depth=2` | Bidirectional citation graph around a paper. Returns nodes and edges for visualisation. |
| `GET /paper/{id}/bridges` | Papers in different fields that are semantically close to this paper — cross-domain connection discovery. |
| `GET /trending` | Top papers by PageRank score (falls back to citation count if scores not computed). |
| `GET /communities` | All detected Louvain communities with labels, top concepts, paper count, and average year. |
| `GET /community/{id}/papers` | Papers belonging to a specific community, ranked by PageRank. |
| `GET /fields` | Field overlap matrix — which top-level fields share the most conceptual vocabulary. |
| `GET /bridge-concepts` | Concepts ranked by how many distinct fields they appear in. The vocabulary of cross-domain thinking. |
| `GET /evolution?concept=...` | How paper count for a concept has changed over time. |
| `GET /cache/stats` | Database health: total papers, abstract coverage, enrichment cache hit rate. |
| `GET /health` | Liveness check. Returns DB status and list of tables present. |

---

## Database Schema

```sql
papers           -- paper_id, title, doi, year, journal, citation_count, abstract
authors          -- author_id, name
paper_authors    -- paper_id, author_id, position
concepts         -- concept_id, name, level (0=field, 1=subfield, 2+= specific)
paper_concepts   -- paper_id, concept_id, score
citations        -- paper_id (citing), cited_paper_id
paper_scores     -- paper_id, pagerank, community, hub_score, authority_score
communities      -- community_id, label, top_concepts, paper_count, avg_year
bridge_concepts  -- concept_id, name, fields, field_count, paper_count, bridge_score
field_overlap    -- field_a, field_b, shared_concepts, overlap_score (Jaccard)
concept_scores   -- concept_id, name, pagerank, field_count, paper_count, bridge_score
concept_evolution-- concept_id, year, paper_count, avg_citations
concept_cooccurrence -- concept_a, concept_b, weight
node_embeddings  -- paper_id, embedding (Node2Vec 128-dim, JSON)
enriched_cache   -- paper_id, fetched_at, success, oa_data (OpenAlex enrichment cache)
```

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/TheNova6000/NexusHub.git
cd NexusHub

python -m venv nexus_env
# Windows:
nexus_env\Scripts\activate
# macOS/Linux:
source nexus_env/bin/activate

pip install -r requirements.txt
```

**Optional dependencies** (degrade gracefully if absent):

```bash
pip install python-louvain   # community detection (Louvain algorithm)
pip install node2vec         # graph embeddings (Node2Vec)
pip install sqlite-vec       # vector search in SQLite
```

---

## Running NexusHub

### Step 1 — Set your OpenAlex email

OpenAlex gives faster, prioritised access to accounts identified by email. Set it once:

```bash
# Windows
set OPENALEX_EMAIL=your@email.com

# macOS/Linux
export OPENALEX_EMAIL=your@email.com
```

### Step 2 — Crawl papers

Start from a set of seed paper IDs (OpenAlex Work IDs, e.g. `W2741809807` for Shannon 1948):

```bash
python discovery.py    # full crawl from seed papers
```

Check coverage at any point:

```bash
python meta_crawl.py --stats
python graph_analysis.py --stats
```

### Step 3 — Resolve ghost nodes (optional, improves graph density)

```bash
python meta_crawl.py --limit 50000   # start with top 50k most-cited ghosts
```

### Step 4 — Generate embeddings

```bash
python embeddings.py   # encode all paper abstracts into vectors
```

### Step 5 — Run graph analysis

```bash
python graph_analysis.py --all
```

This computes PageRank, HITS, communities, bridge concepts, field overlap, and concept evolution. Takes a few minutes on 80k+ papers.

### Step 6 — Start the API

```bash
uvicorn api:app --reload --port 8000
```

The API is at `http://localhost:8000`. Open `index.html` in a browser for the web UI.

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `NEXUS_DB` | `nexushub.db` | Path to the main database |
| `KNOWLEDGE_DB` | `nexushub.db` | Path to the knowledge/embedding database (can be the same file) |
| `OPENALEX_EMAIL` | `nexushub@example.com` | Email for OpenAlex polite pool (faster rate limits) |

---

## Known Limitations

**Ghost paper problem at scale.** OpenAlex contains papers that appear in citation lists but have no full record of their own. At ~80,000 crawled papers, these ghost nodes begin fragmenting the citation graph — the connected component shrinks and PageRank scores become less meaningful. `meta_crawl.py` exists specifically to address this by converting the most-cited ghost nodes into real nodes with lightweight metadata, without the expense of full crawls.

**Concept quality depends on OpenAlex.** The concept taxonomy (fields, subfields, bridge detection) is derived from OpenAlex's automated concept tagging. For older papers or niche fields, concept scores may be sparse or absent.

**Vector search requires `sqlite-vec`.** If `sqlite-vec` is not installed, semantic search is silently disabled and the API falls back to keyword-only search.

---

## Project Structure

```
NexusHub/
├── api.py              # FastAPI application — all REST endpoints, schema-adaptive queries
├── discovery.py        # Citation crawling — builds the initial paper graph from seeds
├── embeddings.py       # Abstract encoding with sentence-transformers, stored via sqlite-vec
├── fix_embeding.py     # Embedding repair utilities (re-encode missing or corrupt embeddings)
├── graph_analysis.py   # Graph analysis — PageRank, HITS, Louvain, Node2Vec, bridge detection
├── knowledge.py        # Knowledge base management utilities
├── meta_crawl.py       # Ghost node resolver — lightweight async metadata crawler
├── index.html          # Web frontend
├── requirements.txt    # Python dependencies
└── run                 # Convenience execution script
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API framework | FastAPI + uvicorn |
| Storage | SQLite (WAL mode) + sqlite-vec |
| Data source | OpenAlex API |
| HTTP client | httpx (async), aiohttp (crawl) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Graph analysis | NetworkX |
| Community detection | python-louvain (Louvain algorithm) |
| Graph embeddings | Node2Vec (128-dim) |

---

## License

MIT License.

---

*Built to understand how knowledge connects — not as a metaphor, but as a measurable, traversable structure.*
