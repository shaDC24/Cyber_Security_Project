"""
attacker/substitute_def.py
---------------------------
The ATTACKER's own substitute ("knockoff") architecture. This is declared
independently on the attacker side: the adversary does NOT know the victim's
true architecture and simply picks one of their own. Here it happens to match
the victim's SmallCNN shape, which is the favourable case Orekondy et al.
study, but nothing in the attack code assumes access to victim/model_def.py --
the attacker only ever talks to the HTTP API.
"""

import torch.nn as nn
import torch.nn.functional as F


class SubstituteCNN(nn.Module):
    """Attacker's chosen substitute: conv(1->16)->conv(16->32)->FC->FC."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
