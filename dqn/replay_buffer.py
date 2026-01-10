import numpy as np
import torch
from pydantic import BaseModel, PositiveInt


class ReplayBufferConfig(BaseModel):
    capacity: PositiveInt = 1_000_000
    agent_history_length: PositiveInt = 4


class ReplayBuffer:
    def __init__(self, config: ReplayBufferConfig, frame_shape: tuple[int]):
        self.capacity = config.capacity
        self.agent_history_length = config.agent_history_length

        self.frames = np.empty((self.capacity, *frame_shape), dtype=np.uint8)
        self.frame_numbers = np.empty(self.capacity, dtype=np.int32)
        self.actions = np.empty(self.capacity, dtype=np.uint8)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=np.bool_)

        self.pos = 0
        self.size = 0
        self.full = False

    def store_initial_frame(self, frame: np.ndarray):
        self.frames[self.pos] = frame
        self.frame_numbers[self.pos] = 0

    def store_experience(
        self,
        action: int,
        reward: float,
        done: bool,
        next_frame: np.ndarray,
        next_frame_number: int,
    ):
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.pos == 0:
            self.full = True

        self.frames[self.pos] = next_frame
        self.frame_numbers[self.pos] = next_frame_number

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        """Randomly select indices from the replay memory.

        Indices selection follows these rules:
            - Avoid picking an index in [pos - history_length, pos] to avoid mixing transitions from different episodes.
            - Avoid picking an index for which s_t+1 does not exist (i.e. the last transition of an episode).
            - No replacement

        Args:
            batch_size: Number of indices to select.

        Returns:
            A numpy array of indices.
        """
        # -1 to avoid picking the last element that does not have transition yet
        upper_bound = self.size - 1
        if upper_bound < batch_size:
            raise ValueError(
                f"Unable to get {batch_size} samples.Replay memory only contains {upper_bound} experiences."
            )

        indices = set()
        for _ in range(batch_size):
        # while len(indices) < batch_size:
            idx = np.random.randint(0, upper_bound)

            if self.full:
                # Avoid picking an index in [pos - history_length, pos]
                if self.pos - self.agent_history_length <= idx <= self.pos:
                    continue

            # No valid s_t+1 for the last episode transition
            if self.dones[idx]:
                continue

            indices.add(idx)

        return np.array(list(indices))

    def _get_batch_states(self, indices: np.ndarray) -> np.ndarray:
        """Get the states for a batch of indices by modifying indices first."""
        batch_size = len(indices)

        # Create array of history indices, e.g [998, 999, 0, 1] for index 1
        offsets = np.arange(self.agent_history_length) - (self.agent_history_length - 1)
        history_indices = (indices[:, None] + offsets) % self.capacity

        # Get array of valid indices by comparing history frame numbers to index frame number
        current_frame_numbers = self.frame_numbers[indices]
        history_frame_numbers = self.frame_numbers[history_indices]
        valid_mask = history_frame_numbers <= current_frame_numbers[:, None]

        # Get the first valid index for each row
        first_valid_relative_idx = np.argmax(valid_mask, axis=1)

        # Get array of first valid frame index for each row
        fill_indices = history_indices[np.arange(batch_size), first_valid_relative_idx]

        # 5. Remplacer TOUS les indices invalides par les fill_indices correspondants
        # Replace all invalid indices by the first valid frame indices
        invalid_mask = ~valid_mask
        filler = np.repeat(fill_indices[:, None], self.agent_history_length, axis=1)
        history_indices[invalid_mask] = filler[invalid_mask]

        states = self.frames[history_indices]
        return states

    def get_experience_samples(
        self, batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly select experiences from the replay memory.

        Args:
            batch_size: Number of experiences to get from the replay memory.

        Raises:
            ValueError: If the replay memory does not contain enough experiences.

        Returns:
            A tuple of 5 elements:
                - Tensor of selected states of shape (batch_size, agent_history_length, height, width)
                - Tensor of selected actions of shape (batch_size, 1)
                - Tensor of selected rewards of shape (batch_size, 1)
                - Tensor of selected dones of shape (batch_size, 1)
                - Tensor of selected next states of shape (batch_size, agent_history_length, height, width)
        """
        indices = self._sample_indices(batch_size)
        next_indices = (indices + 1) % self.capacity

        states = self._get_batch_states(indices)
        next_states = self._get_batch_states(next_indices)

        return (
            torch.from_numpy(states).to(torch.float32),
            torch.from_numpy(self.actions[indices]).to(torch.int32),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.dones[indices]).to(torch.float32),
            torch.from_numpy(next_states).to(torch.float32),
        )
