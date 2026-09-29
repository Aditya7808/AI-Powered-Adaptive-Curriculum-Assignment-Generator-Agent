import os
import sqlite3

# Create dummy db file if not exists
os.makedirs("data", exist_ok=True)
db_path = "data/app_test.db"
if os.path.exists(db_path):
    os.remove(db_path)
conn = sqlite3.connect(db_path)
conn.close()
