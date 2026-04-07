import sqlite3
import sqlite_vec
import struct
import numpy as np
from sentence_transformers import SentenceTransformer

DB="knowledge.db"
MODEL="sentence-transformers/all-MiniLM-L6-v2"

conn=sqlite3.connect(DB)
conn.enable_load_extension(True)
sqlite_vec.load(conn)

model=SentenceTransformer(MODEL)

rows=conn.execute("""
SELECT paper_id,title,abstract
FROM papers
WHERE paper_id NOT IN (
SELECT paper_id FROM paper_embeddings
)
""").fetchall()

print("Missing embeddings:",len(rows))

for pid,title,abstract in rows:

    text=(title or "")+" "+(abstract or "")

    if not text.strip():
        continue

    vec=model.encode(text,normalize_embeddings=True)
    blob=struct.pack(f"{len(vec)}f",*vec)

    conn.execute(
        "INSERT INTO paper_embeddings VALUES (?,?,?,datetime('now'))",
        (pid,blob,MODEL)
    )

    conn.execute(
        "INSERT INTO vec_papers VALUES (?,?)",
        (pid,blob)
    )

conn.commit()
print("Done")