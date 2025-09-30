

import torch.nn as nn


"""Neural network model definitions for myann.

Keep this module minimal: only the model class is defined here so importing
`myann.models` stays lightweight.
"""


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int = 1,
                 hidden: int = 64,
                 out_dim: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        # layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        # layers.append(nn.Dropout(dropout))
        # layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        # layers.append(nn.Dropout(dropout))
        # layers += [nn.Linear(hidden, hidden), nn.Sigmoid()]
        # layers.append(nn.Dropout(dropout))
        # layers.append(nn.Linear(hidden, out_dim))
        # self.net = nn.Sequential(*layers)
        layers = [nn.Linear(in_dim, hidden), nn.Sigmoid()]
        layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.Sigmoid()]
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
