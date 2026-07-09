import os
import sqlite3
import pandas as pd

def setup_database():
    # Check if database already exists
    if os.path.exists("data/olist.db"):
        return
    
    os.makedirs("data", exist_ok=True)
    print("Database not found - please run setup_db.py locally first")

if __name__ == "__main__":
    setup_database()