# scripts/extract_training_data.py

import sqlite3
import pandas as pd

DB_PATH = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\sensor_health.db"

def load_sequences(window_size=10):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM sensor_metrics ORDER BY timestamp ASC", conn)
    conn.close()

    features = ["brightness", "contrast", "blur", "entropy"]
    sequences = []
    for i in range(len(df) - window_size):
        seq = df[features].iloc[i:i+window_size].values
        sequences.append(seq)

    return sequences

# Example: Extract and save sequences to train later
if __name__ == "__main__":
    seqs = load_sequences(window_size=10)
    print(f"[INFO] Extracted {len(seqs)} sequences of 10-frame metrics.")
