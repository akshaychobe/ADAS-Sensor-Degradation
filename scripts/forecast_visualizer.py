# scripts/forecast_visualizer.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_lstm_forecast import ForecastLSTM
from extract_training_data import load_sequences

# --- Load recent data sequences ---
sequences = load_sequences(window_size=10)
X = np.array(sequences[-1:])  # Take the latest sequence for forecasting
X_tensor = torch.tensor(X, dtype=torch.float32)

# --- Load trained model ---
model = ForecastLSTM()
model.load_state_dict(torch.load("models/lstm_forecast.pth"))
model.eval()

# --- Forecast next 10 steps ---
preds = []
input_seq = X_tensor.clone()

for _ in range(10):
    with torch.no_grad():
        next_pred = model(input_seq)
        preds.append(next_pred.numpy())
        # Append prediction to the input sequence, slide window
        next_frame = next_pred.unsqueeze(1)  # shape (B, 1, 4)
        input_seq = torch.cat((input_seq[:, 1:], next_frame), dim=1)

preds = np.vstack(preds)

# --- Visualization ---
metrics = ["Brightness", "Contrast", "Blur", "Entropy"]
plt.figure(figsize=(12, 6))

for i in range(4):
    plt.plot(range(10), preds[:, i], label=metrics[i])

plt.title("Forecasted Sensor Health Metrics for Next 10 Frames")
plt.xlabel("Future Frame Index")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/forecast_visualization.png")
plt.show()
