from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 1, num_actions: int = 5, variant: str = "small") -> None:
        super().__init__()
        if variant == "spatial":
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((3, 3)),
                nn.Flatten(),
                nn.Linear(64 * 3 * 3, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoalCNN(nn.Module):
    """Predicts a goal/frontier cell inside a local square window."""

    def __init__(self, in_ch: int = 6, window: int = 31, hidden: int = 64) -> None:
        super().__init__()
        self.window = int(window)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.encoder(x)
        return logits.flatten(1)

