# trial2_drift_detection.py

import sqlite3
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from forecast_lstm_train import ForecastLSTM, DB_PATH, MODEL_PATH, LOOKBACK

THRESHOLD_MAE = 15.0  # adjustable drift threshold per metric

# === Load latest sequence and next true point ===
def load_sequence_and_label():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT brightness, contrast, blur, entropy FROM sensor_metrics ORDER BY timestamp")
    data = cursor.fetchall()
    conn.close()
    data = np.array(data, dtype=np.float32)
    return data[-(LOOKBACK+1):-1], data[-1]  # X (lookback), y (true next)

# === Forecast next point ===
def forecast_next_point(x_seq, scaler):
    model = ForecastLSTM()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    x_scaled = scaler.transform(x_seq)
    input_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prediction_scaled = model(input_tensor).numpy()[0]  # shape: (4,)

    return scaler.inverse_transform([prediction_scaled])[0]

# === Main Drift Detection ===
def detect_drift():
    x_seq, y_true = load_sequence_and_label()
    scaler = MinMaxScaler()
    scaler.fit(x_seq)

    y_pred = forecast_next_point(x_seq, scaler)
    mae = np.abs(y_true - y_pred)

    print("\n=== Drift Detection Report ===")
    print("True   :", np.round(y_true, 2))
    print("Predicted:", np.round(y_pred, 2))
    print("MAE    :", np.round(mae, 2))

    drift_flags = mae > THRESHOLD_MAE
    labels = ["Brightness", "Contrast", "Blur", "Entropy"]
    for i, flag in enumerate(drift_flags):
        if flag:
            print(f"[ALERT] Drift in {labels[i]} (MAE = {mae[i]:.2f})")
    if not any(drift_flags):
        print("[OK] No significant drift detected.")

if __name__ == "__main__":
    detect_drift()
