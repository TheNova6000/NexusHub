import requests
import sqlite3
import time
import heapq
import math


conn = sqlite3.connect("nexushub.db")
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT,
    doi TEXT,
    year INTEGER,
    journal TEXT,
    citation_count INTEGER,
    abstract TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS authors (
    author_id TEXT PRIMARY KEY,
    name TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id TEXT,
    author_id TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS keywords (
    keyword_id TEXT PRIMARY KEY,
    name TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS paper_keywords (
    paper_id TEXT,
    keyword_id TEXT,
    score REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS related_works (
    paper_id TEXT,
    related_paper_id TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS concepts (
    concept_id TEXT PRIMARY KEY,
    name TEXT,
    level INTEGER,
    wikidata TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS paper_concepts (
    paper_id TEXT,
    concept_id TEXT,
    score REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS citations (
    paper_id TEXT,
    cited_paper_id TEXT
)
""")
cur.execute("""
CREATE TABLE paper_references (
    paper_id TEXT,
    referenced_work_id TEXT
)
""")

cur.execute("CREATE INDEX IF NOT EXISTS idx_citations_paper ON citations(paper_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_citations_ref ON citations(cited_paper_id)")

conn.commit()
def reconstruct_abstract(index):

    if not index:
        return None

    words = {}

    for word, positions in index.items():
        for pos in positions:
            words[pos] = word

    return " ".join(words[i] for i in sorted(words))

nexus_instance = [
"W1995875735",
"W2126160338",
"W1493919779",
"W1995341919",
"W2057236759",
"W2126466006",
"W2008620264",
"W4243566455",
"W1533179050"
]
applicants = set()
visited = set()
paper_cache = {}
print("Started Nexus Hub")

def fetch_paper(pid):

    if pid in paper_cache:
        return paper_cache[pid]

    try:
        url = f"https://api.openalex.org/works/{pid}"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if "id" not in data:
            return None

        paper_cache[pid] = data
        return data

    except:
        return None

def fetch_citing_papers(pid):

    url = f"https://api.openalex.org/works?filter=cites:{pid}&per_page=200"

    try:
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return []

        data = r.json()

        return data.get("results", [])

    except:
        return []

def extraction(seed_id,applicants,visited):
    data = fetch_paper(seed_id)
    if not data:
        print("Failed to fetch:", seed_id)
        return
    pid = data["id"].split("/")[-1]
    visited.add(pid)
    title = data.get("title")
    doi = data.get("doi")
    year = data.get("publication_year")
    citations = data.get("cited_by_count", 0)
    

    journal = None
    if data.get("primary_location") and data["primary_location"].get("source"):
        journal = data["primary_location"]["source"].get("display_name")

    abstract = reconstruct_abstract(
        data.get("abstract_inverted_index")
    )

    cur.execute(
        """
        INSERT OR IGNORE INTO papers
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (pid, title, doi, year, journal, citations, abstract),
    )


    # authors
    for a in data.get("authorships", []):
        author = a.get("author")

        if not author:
            continue
        author_id_raw = author.get("id")
        name = author.get("display_name")
        if not author_id_raw:
            continue

        author_id = author_id_raw.split("/")[-1]

        cur.execute(
            "INSERT OR IGNORE INTO authors VALUES (?, ?)",
            (author_id, name),
        )

        cur.execute(
            "INSERT OR IGNORE INTO paper_authors VALUES (?, ?)",
            (pid, author_id),
        )


    # keywords
    for k in data.get("keywords", []):

        keyword_id = k["id"].split("/")[-1]
        name = k["display_name"]
        score = k["score"]

        cur.execute(
            "INSERT OR IGNORE INTO keywords VALUES (?, ?)",
            (keyword_id, name),
        )

        cur.execute(
            "INSERT OR IGNORE INTO paper_keywords VALUES (?, ?, ?)",
            (pid, keyword_id, score),
        )


    # concepts
    for c in data.get("concepts", []):

        concept_id = c["id"].split("/")[-1]
        name = c["display_name"]
        level = c["level"]
        wikidata = c["wikidata"]
        score = c["score"]

        cur.execute(
            """
            INSERT OR IGNORE INTO concepts
            VALUES (?, ?, ?, ?)
            """,
            (concept_id, name, level, wikidata),
        )

        cur.execute(
            "INSERT OR IGNORE INTO paper_concepts VALUES (?, ?, ?)",
            (pid, concept_id, score),
        )

    citing_papers = fetch_citing_papers(pid)

    for paper in citing_papers:

        cid = paper["id"].split("/")[-1]

        applicants.add(cid)

        cur.execute(
            "INSERT OR IGNORE INTO citations (paper_id, cited_paper_id) VALUES (?, ?)",
            (cid, pid),
        )

    # references
    for r in data.get("referenced_works", []):
        ref_id = r.split("/")[-1]

        applicants.add(ref_id)

        cur.execute(
            "INSERT OR IGNORE INTO paper_references VALUES (?, ?)",
            (pid, ref_id),
        )

    # related works
    for r in data.get("related_works", []):
        rid = r.split("/")[-1]
        applicants.add(rid)
        cur.execute(
            "INSERT OR IGNORE INTO related_works VALUES (?, ?)",
            (pid, rid),
       )
    time.sleep(0.1)



for seed_id in nexus_instance:
    extraction(seed_id,applicants,visited)


print("Starting Nexus Hub Expansion")
#estimation
def estimate_priority(data):

    citations = data.get("cited_by_count", 0)
    concept_count = len(data.get("concepts", []))

    return math.log(citations + 1) + concept_count

#score
def compute_nexus_score(data):

    citations = data.get("cited_by_count", 0)
    concepts = data.get("concepts", [])

    if not concepts:
        return 0

    scores = [c["score"] for c in concepts]

    total = sum(scores)

    if total <= 0:
        return 0

    entropy = 0

    for s in scores:

        p = s / total

        if p <= 0:
            continue

        entropy -= p * math.log(p)

    return math.log(citations + 1) * entropy

print(applicants)

count = 0
for n in applicants.copy():
    data = fetch_paper(n)
    if not data:
        continue
    title = data.get("title")
    if n in visited:
        continue
    print(f"Crawling:{n}:{title}")

    nexus_score = compute_nexus_score(data)
    if nexus_score > 3:
        extraction(n,applicants,visited)
        count +=1
    if count % 10 == 0:
        conn.commit()

    time.sleep(0.1)

    




print(count)
conn.commit()
conn.close()