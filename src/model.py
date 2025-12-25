import torch
import torch.nn as nn


class Hybrid_DeepSwallow(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN: 특징 추출
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # LSTM: 시계열 분석
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        # Classifier
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.permute(0, 2, 1)
        output, (hidden, cell) = self.lstm(x)
        last_memory = hidden[-1]
        return self.fc(last_memory)
