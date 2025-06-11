# trial1_drift_threshold_alert.py
import pandas as pd
from datetime import datetime

# Thresholds (can be tuned later)
THRESHOLDS = {
    "brightness": (30, 200),
    "contrast": (20, 80),
    "blur": (100, float("inf")),
    "entropy": (6.5, float("inf")),
}

# Load the CSV
CSV_PATH = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\image_quality_metrics.csv"
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

# Check for alerts
for _, row in df.iterrows():
    alerts = []
    for metric, (low, high) in THRESHOLDS.items():
        if not (low <= row[metric] <= high):
            alerts.append(f"{metric.upper()} OUT OF RANGE ({row[metric]})")

    if alerts:
        print(f"[ALERT] {row['timestamp']} -> " + " | ".join(alerts))
