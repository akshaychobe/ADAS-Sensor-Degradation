# trial3_fastapi_sensor_health.py
from fastapi import FastAPI
import pandas as pd

CSV_PATH = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\image_quality_metrics.csv"
app = FastAPI()

@app.get("/health")
def sensor_health():
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        return {"status": "No data"}

    latest = df.iloc[-1]
    health = "OK"
    if latest["blur"] < 150 or latest["brightness"] < 30:
        health = "Sensor Degraded"

    return {
        "timestamp": latest["timestamp"],
        "brightness": latest["brightness"],
        "contrast": latest["contrast"],
        "blur": latest["blur"],
        "entropy": latest["entropy"],
        "sensor_status": health
    }
