# forecast_lstm_predict.py

import os
import sqlite3
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from forecast_lstm_train import ForecastLSTM, DB_PATH, MODEL_PATH, LOOKBACK, FORECAST_HORIZON

# === Load data ===
def load_recent_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT brightness, contrast, blur, entropy FROM sensor_metrics ORDER BY timestamp")
    data = cursor.fetchall()
    conn.close()
    return np.array(data, dtype=np.float32)

# === Forecast Future ===
def forecast_recursive(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    recent_sequence = data_scaled[-LOOKBACK:]  # shape: (10, 4)
    input_seq = torch.tensor(recent_sequence, dtype=torch.float32).unsqueeze(0)  # shape: (1, 10, 4)

    model = ForecastLSTM()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    forecasts = []
    with torch.no_grad():
        for _ in range(FORECAST_HORIZON):
            pred = model(input_seq)  # shape: (1, 4)
            forecasts.append(pred.numpy())
            # Slide window: remove oldest, append prediction
            next_input = pred.unsqueeze(1)  # (1, 1, 4)
            input_seq = torch.cat([input_seq[:, 1:], next_input], dim=1)

    forecasts = np.vstack(forecasts)
    return scaler.inverse_transform(forecasts)

# === Plot Results ===
def plot_forecast(past_data, forecast):
    timesteps = np.arange(len(past_data) + len(forecast))
    plt.figure(figsize=(12, 6))
    labels = ["Brightness", "Contrast", "Blur", "Entropy"]

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(timesteps[:len(past_data)], past_data[:, i], label="Past", color="blue")
        plt.plot(timesteps[len(past_data):], forecast[:, i], label="Forecast", color="red")
        plt.title(labels[i])
        plt.legend()

    plt.tight_layout()
    plt.savefig("data/forecast_visualization.png")
    plt.show()

# === Main Execution ===
if __name__ == "__main__":
    data = load_recent_data()
    forecast = forecast_recursive(data)
    plot_forecast(data[-LOOKBACK:], forecast)