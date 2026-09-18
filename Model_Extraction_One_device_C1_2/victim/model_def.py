import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSoftmax(nn.Module):
    """f(x) = softmax(W . flatten(x) + b).  Input: (N, 1, 28, 28) in [0, 1]."""

    def __init__(self, n_classes=10, in_dim=28 * 28):
        super().__init__()
        self.in_dim = in_dim
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits  # CrossEntropyLoss expects raw logits

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class SmallCNN(nn.Module):
    """conv(1->16) -> conv(16->32) -> FC(32*7*7->64) -> FC(64->10)."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
