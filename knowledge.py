import asyncio
import aiohttp
import sqlite3

NEXUS_DB = "nexushub.db"
KNOW_DB = "knowledge.db"

MAX_CONCURRENT_REQUESTS = 10
BATCH_SIZE = 50
count = 0

paper_cache = {}

# ----------------------------
# DATABASE CONNECTIONS
# ----------------------------

nexus_conn = sqlite3.connect(NEXUS_DB)
nexus_cur = nexus_conn.cursor()

knowledge_conn = sqlite3.connect(KNOW_DB)
knowledge_conn.execute("PRAGMA journal_mode=WAL")
cur = knowledge_conn.cursor()

# ----------------------------
# CREATE QUEUE TABLE
# ----------------------------

cur.execute("""
CREATE TABLE IF NOT EXISTS crawl_queue(
    paper_id TEXT PRIMARY KEY,
    status TEXT DEFAULT 'pending'
)
""")

knowledge_conn.commit()

# ----------------------------
# COPY SCHEMA SAFELY
# ----------------------------

def ensure_schema():
    tables = [
        "papers",
        "authors",
        "paper_authors",
        "keywords",
        "paper_keywords",
        "concepts",
        "paper_concepts",
        "citations",
        "paper_references",
        "related_works"
    ]

    for table in tables:

        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        )

        if cur.fetchone():
            continue

        nexus_cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        )

        row = nexus_cur.fetchone()

        if row:
            cur.execute(row[0])

    knowledge_conn.commit()

# ----------------------------
# ABSTRACT RECONSTRUCTION
# ----------------------------

def reconstruct_abstract(index):

    if not index:
        return None

    words = {}

    for word, positions in index.items():
        for pos in positions:
            words[pos] = word

    return " ".join(words[i] for i in sorted(words))

# ----------------------------
# FETCH PAPER
# ----------------------------

async def fetch_paper(session, pid):

    if pid in paper_cache:
        return paper_cache[pid]

    url = f"https://api.openalex.org/works/{pid}"

    try:
        async with session.get(url, timeout=15) as r:

            if r.status != 200:
                return None

            data = await r.json()

            if not data.get("id"):
                return None

            paper_cache[pid] = data
            return data

    except Exception as e:
        print("Fetch error:", pid, e)
        return None

# ----------------------------
# FETCH CITATIONS
# ----------------------------

async def fetch_citing_papers(session, pid):

    url = f"https://api.openalex.org/works?filter=cites:{pid}&per_page=200"

    try:
        async with session.get(url, timeout=15) as r:

            if r.status != 200:
                return []

            data = await r.json()

            return data.get("results", [])

    except:
        return []

# ----------------------------
# STORE DATA
# ----------------------------

def extract_store(data, citing_papers):

    pid_raw = data.get("id")
    if not pid_raw:
        return

    pid = pid_raw.split("/")[-1]

    title = data.get("title")
    doi = data.get("doi")
    year = data.get("publication_year")
    citations = data.get("cited_by_count",0)

    journal = None
    primary = data.get("primary_location") or {}

    source = primary.get("source") if isinstance(primary,dict) else None

    if source:
        journal = source.get("display_name")

    abstract = reconstruct_abstract(
        data.get("abstract_inverted_index")
    )

    cur.execute("""
    INSERT OR IGNORE INTO papers
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,(pid,title,doi,year,journal,citations,abstract))

    # AUTHORS
    for a in data.get("authorships",[]):

        author = a.get("author")
        if not author:
            continue

        aid = author.get("id")
        if not aid:
            continue

        author_id = aid.split("/")[-1]
        name = author.get("display_name")

        cur.execute(
            "INSERT OR IGNORE INTO authors VALUES (?,?)",
            (author_id,name)
        )

        cur.execute(
            "INSERT OR IGNORE INTO paper_authors VALUES (?,?)",
            (pid,author_id)
        )

    # KEYWORDS
    for k in data.get("keywords",[]):

        kid = k.get("id")
        if not kid:
            continue

        kid = kid.split("/")[-1]

        name = k.get("display_name")
        score = k.get("score")

        cur.execute(
            "INSERT OR IGNORE INTO keywords VALUES (?,?)",
            (kid,name)
        )

        cur.execute(
            "INSERT OR IGNORE INTO paper_keywords VALUES (?,?,?)",
            (pid,kid,score)
        )

    # CONCEPTS
    for c in data.get("concepts",[]):

        cid = c.get("id")
        if not cid:
            continue

        cid = cid.split("/")[-1]

        name = c.get("display_name")
        level = c.get("level")
        wikidata = c.get("wikidata")
        score = c.get("score")

        cur.execute("""
        INSERT OR IGNORE INTO concepts
        VALUES (?, ?, ?, ?)
        """,(cid,name,level,wikidata))

        cur.execute(
            "INSERT OR IGNORE INTO paper_concepts VALUES (?,?,?)",
            (pid,cid,score)
        )

    # CITATIONS
    for paper in citing_papers:

        pid_raw = paper.get("id")
        if not pid_raw:
            continue

        cid = pid_raw.split("/")[-1]

        cur.execute(
            "INSERT OR IGNORE INTO citations VALUES (?,?)",
            (cid,pid)
        )

    # REFERENCES
    for r in data.get("referenced_works",[]):

        if not r:
            continue

        ref_id = r.split("/")[-1]

        cur.execute(
            "INSERT OR IGNORE INTO paper_references VALUES (?,?)",
            (pid,ref_id)
        )

    # RELATED
    for r in data.get("related_works",[]):

        if not r:
            continue

        rid = r.split("/")[-1]

        cur.execute(
            "INSERT OR IGNORE INTO related_works VALUES (?,?)",
            (pid,rid))

# ----------------------------
# BUILD QUEUE
# ----------------------------

def build_queue():

    ids = set()

    nexus_cur.execute("SELECT paper_id FROM citations")
    ids.update(x[0] for x in nexus_cur.fetchall())

    nexus_cur.execute("SELECT cited_paper_id FROM citations")
    ids.update(x[0] for x in nexus_cur.fetchall())

    nexus_cur.execute("SELECT referenced_work_id FROM paper_references")
    ids.update(x[0] for x in nexus_cur.fetchall())

    nexus_cur.execute("SELECT related_paper_id FROM related_works")
    ids.update(x[0] for x in nexus_cur.fetchall())

    print("Total candidate papers:",len(ids))

    added = 0

    for pid in ids:

        nexus_cur.execute(
            "SELECT 1 FROM papers WHERE paper_id=?",
            (pid,)
        )

        if nexus_cur.fetchone():
            continue

        cur.execute(
            "INSERT OR IGNORE INTO crawl_queue VALUES (?, 'pending')",
            (pid,)
        )

        added += 1

    knowledge_conn.commit()

    print("Added to queue:",added)

# ----------------------------
# LOAD BATCH
# ----------------------------

def load_batch():

    cur.execute("""
    SELECT paper_id
    FROM crawl_queue
    WHERE status='pending'
    LIMIT ?
    """,(BATCH_SIZE,))

    rows = cur.fetchall()

    ids = [r[0] for r in rows]

    for pid in ids:

        cur.execute(
            "UPDATE crawl_queue SET status='processing' WHERE paper_id=?",
            (pid,)
        )

    knowledge_conn.commit()

    return ids

# ----------------------------
# MARK DONE
# ----------------------------

def mark_done(pid):

    cur.execute(
        "UPDATE crawl_queue SET status='done' WHERE paper_id=?",
        (pid,)
    )

# ----------------------------
# PROCESS BATCH
# ----------------------------

async def process_batch(batch):

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(connector=connector) as session:

        try:

            tasks = [fetch_paper(session,p) for p in batch]
            papers = await asyncio.gather(*tasks, return_exceptions=True)

            citation_tasks = []

            for p in papers:

                if not p or isinstance(p,Exception):
                    citation_tasks.append(asyncio.sleep(0))
                else:

                    pid = p.get("id","").split("/")[-1]

                    citation_tasks.append(
                        fetch_citing_papers(session,pid)
                    )

            citations = await asyncio.gather(*citation_tasks, return_exceptions=True)

            return papers,citations

        except asyncio.CancelledError:
            print("Batch cancelled safely")
            return [],[]

# ----------------------------
# MAIN LOOP
# ----------------------------

async def run():

    while True:

        batch = load_batch()
    
        if not batch:
            print("Crawl finished")
            break

        print( "Processing batch:",len(batch))

        papers,citations = await process_batch(batch)

        for data,citing,pid in zip(papers,citations,batch):

            if not data or isinstance(data,Exception):
                continue

            extract_store(data,citing)
            mark_done(pid)

        knowledge_conn.commit()

# ----------------------------
# START
# ----------------------------

if __name__ == "__main__":

    ensure_schema()

    build_queue()

    asyncio.run(run())