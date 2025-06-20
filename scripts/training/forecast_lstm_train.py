# forecast_lstm_train.py

import os
import sqlite3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# === Hyperparameters ===
DB_PATH = r"C:\\Users\\Lenovo\\ADAS-Sensor-Degradation\\data\\sensor_health.db"
LOOKBACK = 10
FORECAST_HORIZON = 10
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.001
MODEL_PATH = "models/lstm_forecaster.pth"

# === Dataset Loader ===
class SensorDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# === LSTM Model ===
class ForecastLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])  # Single-step prediction

# === Helper: Load data from SQLite ===
def load_sensor_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT brightness, contrast, blur, entropy FROM sensor_metrics ORDER BY timestamp")
    data = cursor.fetchall()
    conn.close()
    return np.array(data, dtype=np.float32)

# === Helper: Create sequences ===
def create_sequences(data, lookback=LOOKBACK):
    sequences = []
    for i in range(len(data) - lookback - 1):
        x_seq = data[i:i+lookback]
        y_seq = data[i+lookback]  # Only 1 step ahead
        sequences.append((x_seq, y_seq))
    return sequences


# === Main Training Loop ===
def train():
    data = load_sensor_data()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    sequences = create_sequences(data)
    train_loader = DataLoader(SensorDataset(sequences), batch_size=BATCH_SIZE, shuffle=True)

    model = ForecastLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        losses = []
        for x_batch, y_batch in train_loader:
            output = model(x_batch)
            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {np.mean(losses):.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[SAVED] Model -> {MODEL_PATH}")

if __name__ == "__main__":
    train()
