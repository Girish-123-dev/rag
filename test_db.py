"""
Quick database connection and content verification test.
Run this anytime to confirm pgvector is working correctly.

Usage:
    python test_db.py
"""
import psycopg2

conn = psycopg2.connect('postgresql://capstone_user:root@localhost:5432/capstone_db')
cur = conn.cursor()

# 1. Check pgvector extension
cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
version = cur.fetchone()
print(f"✅ pgvector version:      {version[0]}")

# 2. Check collection exists
cur.execute("SELECT name FROM langchain_pg_collection WHERE name = 'dementia_documents'")
collection = cur.fetchone()
print(f"✅ Collection:            {collection[0] if collection else 'NOT FOUND'}")

# 3. Check chunk count
cur.execute("""
    SELECT COUNT(*) FROM langchain_pg_embedding e
    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
    WHERE c.name = 'dementia_documents'
""")
count = cur.fetchone()
print(f"✅ Chunks in pgvector:    {count[0]}")

# 4. Show a sample chunk so we can confirm content is readable
cur.execute("""
    SELECT e.document FROM langchain_pg_embedding e
    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
    WHERE c.name = 'dementia_documents'
    LIMIT 1
""")
sample = cur.fetchone()
print(f"✅ Sample chunk preview:  {sample[0][:120]}...")

conn.close()
print("\n🎉 Database is healthy and ready!")