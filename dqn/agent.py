from dqn.model import Model
from dqn.replay_buffer import ReplayBuffer

import gymnasium as gym
import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

writer = SummaryWriter()


@torch.jit.script
def fused_target_q_calculation(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    next_state_max_q_values: torch.Tensor,
) -> torch.Tensor:
    return (
        rewards.unsqueeze(dim=1)
        + (1 - dones.unsqueeze(dim=1)) * gamma * next_state_max_q_values
    )


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
    ) -> None:
        self.warmup_steps = config.warmup_steps
        self.train_steps = config.steps - self.warmup_steps
        self.update_frequency = config.update_frequency
        self.target_update_frequency = config.target_update_frequency

        self.env = env
        self.replay_buffer = replay_buffer
        self.model = model
        self.target_model = target_model
        self.device = device

        self.num_envs = getattr(env, "num_envs", 1)
        self.learning_steps_counter = 0
        self.episode_rewards = np.zeros(self.num_envs)
        self.state = self._set_initial_state()

        self.loss_fn = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=2.5e-4)
        self.batch_size = config.batch_size
        self.gamma = config.gamma

        self.episode_count = 0
        self.max_frame_numbers = np.zeros(self.num_envs)

    def _set_initial_state(self) -> np.ndarray:
        """Reset the agent's environment and store initial frames in the replay buffer."""
        state, _ = self.env.reset()
        # next_frame is expected to be (num_envs, 84, 84)
        self.replay_buffer.store_initial_frame(state[:, -1])
        return state

    def experiment(
        self, warming_up: bool = False
    ) -> None:
        """Make the agent act on its environment as many times as self.udpate_frequency
        and store state transitions in the replay buffer.

        Args:
            warming_up: If set to True, actions are sampled from the environment action space. 
                Otherwise, the model policy is used.Defaults to False.
        """
        for _ in range(self.update_frequency):
            if warming_up:
                actions = self.env.action_space.sample()
            else:
                state_tensor = torch.from_numpy(self.state).to(self.device).byte()
                actions = self.model.policy(state_tensor)
            
            new_state, rewards, dones, truncated, infos = self.env.step(actions)
            
            # In AsyncVectorEnv, infos stores arrays for the batch
            episode_frame_numbers = infos["episode_frame_number"]
            
            # Store experience batch (we take the last frame of the stack)
            self.replay_buffer.store_experience(
                actions, rewards, dones, new_state[:, -1], episode_frame_numbers
            )
            
            self.episode_rewards += rewards
            self.max_frame_numbers = np.maximum(self.max_frame_numbers, episode_frame_numbers)

            # Check for episodes that ended
            for i in range(self.num_envs):
                if dones[i] or truncated[i]:
                    if not warming_up:
                        self.episode_count += 1
                        writer.add_scalar(
                            "Episode reward", self.episode_rewards[i], self.episode_count
                        )
                        writer.add_scalar(
                            "Maximum episode frame number",
                            self.max_frame_numbers[i],
                            self.episode_count,
                        )
                    self.episode_rewards[i] = 0
                    self.max_frame_numbers[i] = 0
            
            self.state = new_state

    def warmup(self) -> None:
        """Warm up the agent by performing random actions."""
        device = self.device
        self.device = torch.device("cpu")
        steps_per_experiment = self.update_frequency * self.num_envs
        with tqdm(total=self.warmup_steps, desc="Warming up") as pbar:
            for _ in range(0, self.warmup_steps, steps_per_experiment):
                self.experiment(warming_up=True)
                pbar.update(steps_per_experiment)
        self.device = device

    def _compute_target_q_values(
        self, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        next_state_max_q_values = self.target_model(next_states).max(
            dim=1, keepdim=True
        )[0]
        target_q_values = fused_target_q_calculation(
            rewards, dones, self.gamma, next_state_max_q_values
        )
        return target_q_values

    def _learn(self) -> torch.Tensor:
        """Perform a single learning step."""
        with torch.no_grad():
            current_states, actions, rewards, dones, next_states = (
                self.replay_buffer.get_experience_samples(self.batch_size)
            )
            current_states = current_states.to(self.device, non_blocking=True, dtype=torch.float32) / 255
            actions = actions.to(self.device, non_blocking=True, dtype=torch.int32)
            rewards = rewards.to(self.device, non_blocking=True, dtype=torch.float32)
            dones = dones.to(self.device, non_blocking=True, dtype=torch.float32)
            next_states = next_states.to(self.device, non_blocking=True, dtype=torch.float32) / 255
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
        steps_per_experiment = self.update_frequency * self.num_envs
        with tqdm(total=self.train_steps, desc="Training") as pbar:
            for _ in range(0, self.train_steps, steps_per_experiment):
                self.experiment()
                self._learn()
                pbar.update(steps_per_experiment)
                self.learning_steps_counter += 1
                if self.learning_steps_counter % self.target_update_frequency == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
