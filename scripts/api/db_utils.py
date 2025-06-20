# db_utils.py
import sqlite3
from datetime import datetime

DB_PATH = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\sensor_health.db"

def create_table():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_id TEXT,
            brightness REAL,
            contrast REAL,
            blur REAL,
            entropy REAL
        )
        """)
        conn.commit()

def insert_metrics(image_id, brightness, contrast, blur, entropy):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO sensor_metrics (timestamp, image_id, brightness, contrast, blur, entropy)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, image_id, brightness, contrast, blur, entropy))
        conn.commit()
