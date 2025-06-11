# scripts/one_time.py

import sqlite3
import numpy as np

DB_PATH = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\sensor_health.db"

def load_sensor_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT brightness, contrast, blur, entropy FROM sensor_metrics ORDER BY timestamp")
    data = cursor.fetchall()
    conn.close()
    return np.array(data, dtype=np.float32)

# Load data
data = load_sensor_data()

# Print shape and variance
print(f"Loaded {len(data)} rows of sensor data.")
print("Variance per metric (brightness, contrast, blur, entropy):")
print(np.var(data, axis=0))
