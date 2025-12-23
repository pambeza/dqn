import logging

from dqn.model import Model
from dqn.replay_buffer import ReplayBuffer

import gymnasium as gym
import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

writer = SummaryWriter()


class AgentConfig(BaseModel):
    batch_size: int = 32
    gamma: float = 0.99
    steps: int = 50_000_000
    target_update_frequency: int = 10_000
    update_frequency: int = 4
    warmup_steps: int = 50_000


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        env: gym.Env,
        replay_buffer: ReplayBuffer,
        model: Model,
        target_model: Model,
        device: torch.device,
    ):
        self.warmup_steps = config.warmup_steps
        self.train_steps = config.steps - self.warmup_steps
        self.update_frequency = config.update_frequency
        self.target_update_frequency = config.target_update_frequency

        self.env = env
        self.replay_buffer = replay_buffer
        self.model = model
        self.target_model = target_model
        self.device = device

        self.learning_steps_counter = 0
        self.state = self._set_initial_state()

        self.loss_fn = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=2.5e-4)
        self.batch_size = config.batch_size
        self.gamma = config.gamma

        self.episode_count = 0
        self.episode_reward = 0

    def _set_initial_state(self) -> np.ndarray:
        state, _ = self.env.reset()
        self.replay_buffer.store_initial_frame(state[-1:])
        return state

    def experiment(
        self, warming_up: bool = False, device: torch.device = "cpu"
    ) -> None:
        for _ in range(self.update_frequency):
            state = torch.from_numpy(self.state).to(device)
            if warming_up:
                action = self.env.action_space.sample()
            else:
                action = self.model.policy(state)
            new_state, reward, done, truncated, info = self.env.step(action)
            self.replay_buffer.store_experience(
                action, reward, done, new_state[-1:], info["episode_frame_number"]
            )
            self.episode_reward += reward
            if done:
                new_state = self._set_initial_state()
                if not warming_up:
                    self.episode_count += 1
                    writer.add_scalar(
                        "Episode reward", self.episode_reward, self.episode_count
                    )
                    self.episode_reward = 0
            self.state = new_state

    def warmup(self) -> None:
        """Warm up the agent by performing random actions."""
        with tqdm(total=self.warmup_steps) as pbar:
            for _ in range(0, self.warmup_steps, self.update_frequency):
                self.experiment(warming_up=True, device=self.device)
                pbar.update(self.update_frequency)

    def _compute_target_q_values(self, next_states, rewards, dones) -> torch.Tensor:
        next_state_max_q_values = self.target_model(next_states).max(
            dim=1, keepdim=True
        )[0]
        target_q_values = (
            rewards.unsqueeze(dim=1)
            + (1 - dones.unsqueeze(dim=1)) * self.gamma * next_state_max_q_values
        )
        return target_q_values

    def _learn(self) -> torch.Tensor:
        """Perform a single learning step."""
        with torch.no_grad():
            current_states, actions, rewards, dones, next_states = (
                self.replay_buffer.get_experience_samples(self.batch_size)
            )
            current_states = current_states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)
            next_states = next_states.to(self.device)
            target_q_values = self._compute_target_q_values(next_states, rewards, dones)

        outputs = self.model(current_states)
        current_state_q_values = outputs.gather(dim=1, index=actions.unsqueeze(dim=1))

        loss = self.loss_fn(target_q_values, current_state_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss

    def train(self) -> None:
        """Train the agent."""
        self.model.to(self.device)
        self.target_model.to(self.device)
        with tqdm(total=self.train_steps) as pbar:
            for _ in range(0, self.train_steps, self.update_frequency):
                self.experiment(device=self.device)
                loss = self._learn()
                pbar.update(self.update_frequency)
                pbar.set_postfix(
                    loss=loss.item(),
                    update=f"{self.learning_steps_counter}/{self.train_steps}",
                )
                self.learning_steps_counter += 1
                if self.learning_steps_counter % self.target_update_frequency == 0:
                    logging.debug("Updating target model weights")
                    self.target_model.load_state_dict(self.model.state_dict())
        torch.save(self.model.state_dict(), "dqn.pt")
