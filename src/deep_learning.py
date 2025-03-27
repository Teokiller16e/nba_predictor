# deep_learning.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


# ---------------------------
# Simple model architectures
# ---------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # x: [batch, seq_len, features]
        out = self.dropout(out[:, -1, :])  # Last timestep
        return torch.sigmoid(self.fc(out))


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return torch.sigmoid(self.fc(out))


class Conv1DModel(nn.Module):
    def __init__(self, input_size, num_filters=64, kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1  # ✅ preserves output width
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # → [batch, features, seq_len]
        x = F.relu(self.conv(x))  # → [batch, filters, seq_len]
        x = self.pool(x).squeeze(-1)  # → [batch, filters]
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))



# ---------------------------
# Simple training wrapper
# ---------------------------

class TorchClassifier:
    def __init__(self, model_class, input_size, epochs=10, batch_size=32, lr=1e-3, device=None, **model_kwargs):
        self.model_class = model_class
        self.input_size = input_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 🆕 Pass additional model-specific parameters (e.g., hidden_size) to the model constructor
        self.model = self.model_class(input_size=input_size, **model_kwargs).to(self.device)

        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def fit(self, X, y):
        X_tensor = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)

        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)

        X_tensor = X_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)

        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X_tensor = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        X_tensor = X_tensor.to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).squeeze().cpu().numpy()
        return (preds >= 0.5).astype(int)
