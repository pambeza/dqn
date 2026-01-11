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

        self.frames = torch.empty((self.capacity, *frame_shape), dtype=torch.uint8)
        self.frame_numbers = torch.empty(self.capacity, dtype=torch.int32)
        self.actions = torch.empty(self.capacity, dtype=torch.uint8)
        self.rewards = torch.empty(self.capacity, dtype=torch.float32)
        self.dones = torch.empty(self.capacity, dtype=torch.bool)

        self.pos = 0
        self.size = 0
        self.full = False

    def store_initial_frame(self, frame: np.ndarray):
        num_envs = frame.shape[0]
        # For simplicity in vectorized envs, we assume we store the initial frame 
        # for all envs at once or we don't use this method as much.
        # However, we need to handle the case where frame is (num_envs, 1, 84, 84)
        indices = (self.pos + torch.arange(num_envs)) % self.capacity
        self.frames[indices] = torch.from_numpy(frame)
        self.frame_numbers[indices] = 0
        # self.pos is NOT updated here normally in the original code, 
        # but in vectorized envs we might need to be careful.
        # The original code only stored ONE frame.

    def store_experience(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_frames: np.ndarray,
        next_frame_numbers: np.ndarray,
    ):
        num_envs = actions.shape[0]
        # Calculate indices for the whole batch
        indices = (self.pos + torch.arange(num_envs)) % self.capacity
        
        self.actions[indices] = torch.from_numpy(actions).byte()
        self.rewards[indices] = torch.from_numpy(rewards).float()
        self.dones[indices] = torch.from_numpy(dones).bool()
        self.frames[indices] = torch.from_numpy(next_frames)
        self.frame_numbers[indices] = torch.from_numpy(next_frame_numbers).int()

        self.pos = (self.pos + num_envs) % self.capacity
        self.size = min(self.size + num_envs, self.capacity)
        if self.size == self.capacity:
            self.full = True

    def _sample_indices(self, batch_size: int) -> torch.Tensor:
        """Randomly select indices from the replay memory using vectorization."""
        upper_bound = self.size - 1
        if upper_bound < batch_size:
            raise ValueError(
                f"Unable to get {batch_size} samples. Replay memory only contains {upper_bound} experiences."
            )

        # Sample more than needed to account for invalid indices (dones, etc.)
        candidate_indices = torch.randint(0, upper_bound, (batch_size * 2,))
        
        # 1. Mask out s_t that are "dones" (no s_t+1 exists)
        valid_mask = ~self.dones[candidate_indices]
        
        # 2. Mask out indices in the current transition range (to avoid mixing different episodes)
        if self.full:
            # Check for range [pos - history, pos] including circular wrap
            within_pos_range = (candidate_indices >= (self.pos - self.agent_history_length)) & (candidate_indices <= self.pos)
            if self.pos < self.agent_history_length:
                # Handle wrap around at the end of the buffer
                within_pos_range |= (candidate_indices >= (self.capacity - (self.agent_history_length - self.pos)))
            valid_mask &= ~within_pos_range
            
        valid_indices = candidate_indices[valid_mask]
        
        # If we didn't get enough, just take what we need from a fresh random sample 
        # (This is a rare fallback)
        if len(valid_indices) < batch_size:
            return torch.randint(0, upper_bound, (batch_size,))
            
        return valid_indices[:batch_size]

    def _get_batch_states(self, indices: np.ndarray) -> np.ndarray:
        """Get the states for a batch of indices by modifying indices first."""
        batch_size = len(indices)

        # Create array of history indices, e.g [998, 999, 0, 1] for index 1
        offsets = torch.arange(self.agent_history_length) - (self.agent_history_length - 1)
        history_indices: torch.Tensor = (indices[:, None] + offsets) % self.capacity

        # Get array of valid indices by comparing history frame numbers to index frame number
        current_frame_numbers = self.frame_numbers[indices]
        history_frame_numbers = self.frame_numbers[history_indices]
        valid_mask = history_frame_numbers <= current_frame_numbers[:, None]

        # Get the first valid index for each row
        first_valid_relative_idx = torch.argmax(valid_mask.to(torch.uint8), dim=1)

        # Get array of first valid frame index for each row
        fill_indices = history_indices[torch.arange(batch_size), first_valid_relative_idx]

        # Replace all invalid indices by the first valid frame indices
        invalid_mask = ~valid_mask
        filler = torch.repeat_interleave(fill_indices[:, None], self.agent_history_length, dim=1)
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
        return states, self.actions[indices], self.rewards[indices], self.dones[indices], next_states