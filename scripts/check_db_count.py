# scripts/check_db_count.py

import sqlite3

DB_PATH = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\sensor_health.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM sensor_metrics;")
count = cursor.fetchone()[0]
conn.close()

print(f"Total rows in sensor_metrics: {count}")
