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


class SubstituteMLP(nn.Module):
    """A DELIBERATELY DIFFERENT attacker architecture: a plain fully-connected
    multilayer perceptron (784 -> 256 -> 128 -> 10), no convolutions at all.

    The victim (SmallCNN) is convolutional; this substitute is not. Using it
    demonstrates the core Knockoff-Nets claim (Orekondy et al. 2019, Sec. 6.2 /
    Fig. 10): functionality can be stolen even when the attacker's architecture
    does NOT match the victim's, because the attack only relies on (input,
    predicted-label) pairs from the API -- never on the victim's structure.
    """

    def __init__(self, n_classes=10, in_dim=28 * 28):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# Registry so experiments can pick a substitute by name without the attack code
# hard-coding one architecture. The attacker owns this list; it is independent
# of anything on the victim side.
SUBSTITUTES = {
    "cnn": SubstituteCNN,
    "mlp": SubstituteMLP,
}
