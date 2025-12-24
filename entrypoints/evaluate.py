from argparse import ArgumentParser
from pathlib import Path

from dqn.model import Model
from dqn.scheduler import ConstantScheduler

import ale_py
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)

# Register envs
gym.register_envs(ale_py)

parser = ArgumentParser(prog="DQN", description="Evaluate DQN.")
parser.add_argument(
    "--model", "-m", type=Path, dest="model_path", help="Path to the model file."
)
parser.add_argument(
    "--steps",
    "-s",
    type=int,
    dest="steps",
    help="Number of steps of play.",
    default=1000,
)
parser.add_argument(
    "--device",
    "-d",
    type=str,
    dest="device",
    help="Device to run the model on.",
    default="cpu",
)


def main(model_path: Path, steps: int, device: str):
    scheduler = ConstantScheduler(0.05)
    model = Model(4, scheduler)
    model.load_state_dict(
        torch.load(model_path.as_posix(), map_location=torch.device(device))
    )
    model.eval()

    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStackObservation(env, stack_size=4)

    state, _ = env.reset()

    for _ in range(steps):
        state_tensor = (
            torch.from_numpy(np.array(state)).to(torch.float32).unsqueeze(dim=0)
        )
        q_values = model(state_tensor)
        action = q_values.argmax(dim=1).item()
        state, reward, done, truncated, _ = env.step(action)

        if done or truncated:
            state, _ = env.reset()

    env.close()


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
