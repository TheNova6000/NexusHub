# import sqlite3
# import sqlite_vec
# import struct
# import numpy as np

# DB_PATH = "nexushub.db"
# DIM = 384


# def connect():
#     conn = sqlite3.connect(DB_PATH)
#     conn.enable_load_extension(True)
#     sqlite_vec.load(conn)
#     conn.enable_load_extension(False)
#     return conn


# def ensure_vec_table(conn):
#     conn.execute(f"""
#         CREATE VIRTUAL TABLE IF NOT EXISTS vec_papers
#         USING vec0(
#             paper_id TEXT PRIMARY KEY,
#             embedding float[{DIM}]
#         );
#     """)
#     conn.commit()


# def main():
#     conn = connect()
#     ensure_vec_table(conn)

#     print("Loading embeddings...")

#     rows = conn.execute(
#         "SELECT paper_id, embedding FROM paper_embeddings"
#     ).fetchall()

#     print(f"Found {len(rows)} embeddings")

#     inserted = 0

#     for pid, blob in rows:
#         conn.execute(
#             "INSERT OR REPLACE INTO vec_papers (paper_id, embedding) VALUES (?, ?)",
#             (pid, blob)
#         )

#         inserted += 1

#         if inserted % 5000 == 0:
#             conn.commit()
#             print(f"Inserted {inserted}/{len(rows)}")

#     conn.commit()

#     total = conn.execute(
#         "SELECT COUNT(*) FROM vec_papers"
#     ).fetchone()[0]

#     print("\nVector index rebuilt.")
#     print("vec_papers rows:", total)

#     conn.close()


# if __name__ == "__main__":
#     main()










import sqlite3
import sqlite_vec
import struct
import numpy as np

conn = sqlite3.connect("nexushub.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)

rows = conn.execute("SELECT COUNT(*) FROM vec_papers_rowids").fetchone()

print("Vectors stored:", rows[0])

row = conn.execute(
    "SELECT embedding FROM paper_embeddings LIMIT 1"
).fetchone()

blob = row[0]

res = conn.execute("""
SELECT paper_id, distance
FROM vec_papers
WHERE embedding MATCH ?
AND k = 5
""", (blob,)).fetchall()

print(res)