import pandas as pd
import sqlite3
from sqlalchemy import create_engine

# Load from PostgreSQL
engine = create_engine("postgresql://sankalpsingh@localhost:5432/olist_db")

print("Loading master_orders from PostgreSQL...")
df = pd.read_sql("SELECT * FROM master_orders", engine)
print(f"Loaded {len(df)} rows")

# Save to SQLite
conn = sqlite3.connect("data/olist.db")
df.to_sql("master_orders", conn, if_exists="replace", index=False)
conn.close()

print("SQLite database created at data/olist.db")