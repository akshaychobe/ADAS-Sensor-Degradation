# simulate_degradation_data.py

import sqlite3
import time
import random
from datetime import datetime

DB_PATH = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\sensor_health.db"

# === Function to insert simulated degraded data ===
def insert_fake_row():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Simulated degraded values
    brightness = random.uniform(60, 90)              # lower brightness
    contrast = random.uniform(30, 45)                # lower contrast
    blur = random.uniform(80, 150)                   # increasing blur = degradation
    entropy = random.uniform(6.0, 7.0)               # less detail = degradation

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO sensor_metrics (timestamp, image_id, brightness, contrast, blur, entropy)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, f"simulated_{timestamp}", brightness, contrast, blur, entropy))

    conn.commit()
    conn.close()
    print(f"[INSERTED] {timestamp} | blur={blur:.2f} | contrast={contrast:.2f}")


# === Loop ===
if __name__ == "__main__":
    while True:
        insert_fake_row()
        time.sleep(10)  # simulate new frame every 10 seconds