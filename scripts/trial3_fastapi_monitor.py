# trial3_fastapi_monitor.py

from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scripts.forecast_lstm_train import ForecastLSTM, DB_PATH, MODEL_PATH, LOOKBACK
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="ADAS Sensor Drift Monitor API")

# === Prometheus Drift Gauges ===
brightness_drift = Gauge("sensor_brightness_drift", "Brightness MAE")
contrast_drift = Gauge("sensor_contrast_drift", "Contrast MAE")
blur_drift = Gauge("sensor_blur_drift", "Blur MAE")
entropy_drift = Gauge("sensor_entropy_drift", "Entropy MAE")

# === Constants ===
THRESHOLD_MAE = 15.0
LABELS = ["brightness", "contrast", "blur", "entropy"]

class DriftResult(BaseModel):
    true_values: dict
    predicted_values: dict
    mae_values: dict
    drift_flags: dict
    overall_drift: bool

# === Core Logic ===
def load_sequence_and_label():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT brightness, contrast, blur, entropy FROM sensor_metrics ORDER BY timestamp")
    data = cursor.fetchall()
    conn.close()
    data = np.array(data, dtype=np.float32)
    return data[-(LOOKBACK+1):-1], data[-1]  # X and true y

def forecast_next(x_seq):
    scaler = MinMaxScaler()
    scaler.fit(x_seq)
    x_scaled = scaler.transform(x_seq)
    input_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

    model = ForecastLSTM()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()[0]
    pred = scaler.inverse_transform([pred_scaled])[0]
    return pred

@app.get("/drift", response_model=DriftResult)
def detect_drift():
    x_seq, y_true = load_sequence_and_label()
    y_pred = forecast_next(x_seq)
    mae = np.abs(y_true - y_pred)

    # === Update Prometheus Gauges ===
    brightness_drift.set(float(mae[0]))
    contrast_drift.set(float(mae[1]))
    blur_drift.set(float(mae[2]))
    entropy_drift.set(float(mae[3]))

    drift_flags = mae > THRESHOLD_MAE
    drift_result = DriftResult(
        true_values={LABELS[i]: float(y_true[i]) for i in range(4)},
        predicted_values={LABELS[i]: float(y_pred[i]) for i in range(4)},
        mae_values={LABELS[i]: float(mae[i]) for i in range(4)},
        drift_flags={LABELS[i]: bool(drift_flags[i]) for i in range(4)},
        overall_drift=bool(np.any(drift_flags))
    )

    return drift_result

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
