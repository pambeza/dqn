import numpy as np
from pydantic import BaseModel, PositiveInt
import torch


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

    def store_experience(self, action: int, reward: float, done: bool, next_frame: np.ndarray, next_frame_number: int):
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
        while len(indices) < batch_size:
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

    def _get_state(self, index: int) -> np.ndarray:
        """Get the state at the given index."""
        current_frame_number = self.frame_numbers[index]
        state = []
        for i in range(self.agent_history_length - 1, -1, -1):
            prev_idx = (index - i) % self.capacity
            # Previous frame is actually a future frame
            if self.frame_numbers[prev_idx] > current_frame_number:
                state.append(self.frames[index])
            else:
                state.append(self.frames[prev_idx])

        state = np.stack(state, axis=0)
        return state

    def get_experience_samples(self, batch_size: int = 32):
        """Randomly select experiences from the replay memory.

        Args:
            batch_size: Number of experiences to get from the replay memory.

        Raises:
            ValueError: If the replay memory does not contain enough experiences.

        Returns:
            A tuple of 5 elements:
                - Numpy array of selected states
                - Numpy array of selected actions
                - Numpy array of selected rewards
                - Numpy array of selected dones
                - Numpy array of selected next states
        """
        indices = self._sample_indices(batch_size)
        states = []
        next_states = []
        for idx in indices:
            states.append(self._get_state(idx))
            next_state_idx = (idx + 1) % self.capacity
            next_states.append(self._get_state(next_state_idx))

        return (
            torch.from_numpy(np.array(states)).to(torch.float32),
            torch.from_numpy(self.actions[indices]).to(torch.int32),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.dones[indices]).to(torch.uint8),
            torch.from_numpy(np.array(next_states)).to(torch.float32),
        )
