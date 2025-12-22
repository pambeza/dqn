import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dqn.scheduler import EpsilonScheduler



class Model(nn.Module):
    def __init__(
        self,
        nb_valid_actions: int,
        epsilon_scheduler: EpsilonScheduler,
    ) -> None:
        super().__init__()
        self.nb_valid_actions = nb_valid_actions
        self.frames_count = 0
        self.epsilon_scheduler = epsilon_scheduler
        self.conv_1 = nn.Conv2d(
            in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4
        )
        self.conv_2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2
        )
        self.conv_3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1
        )
        self.linear_1 = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.linear_2 = nn.Linear(in_features=512, out_features=self.nb_valid_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))

        x = x.flatten(start_dim=1)
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

    @torch.no_grad()
    def policy(self, state: torch.Tensor) -> int:
        """Select an action based on the policy."""
        if np.random.rand() < self.epsilon_scheduler.compute_epsilon(self.frames_count):
            action = np.random.randint(self.nb_valid_actions)
        else:
            state = state.unsqueeze(dim=0).to(torch.float32)
            q_values = self.forward(state)
            action = q_values.argmax(dim=1).item()
        self.frames_count += 1
        return action