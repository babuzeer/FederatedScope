import torch
import torch.nn as nn


class BadPFLFeatureGenerator(nn.Module):
    """Feature-space generator used by Bad-PFL on top of GGEUR embeddings."""

    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = max(int(hidden_dim), 32)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)
