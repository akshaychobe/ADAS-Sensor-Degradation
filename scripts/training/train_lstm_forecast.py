# scripts/train_lstm_forecast.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from extract_training_data import load_sequences
import numpy as np

# Load and prepare data
sequences = load_sequences()
X = np.array(sequences[:-1])
y = np.array([s[-1] for s in sequences[1:]])

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define LSTM
class ForecastLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Predict next timestep

model = ForecastLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "models/lstm_forecast.pth")
