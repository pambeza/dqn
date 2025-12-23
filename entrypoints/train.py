import logging
from argparse import ArgumentParser
from pathlib import Path

from dqn.agent import Agent, AgentConfig
from dqn.model import Model
from dqn.replay_buffer import ReplayBuffer, ReplayBufferConfig
from dqn.scheduler import LinearEpsilonScheduler, SchedulerConfig

import ale_py
import gymnasium as gym
import torch
import yaml
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from pydantic import BaseModel

gym.register_envs(ale_py)

logging.basicConfig(level=logging.DEBUG)


class TrainConfig(BaseModel):
    scheduler: SchedulerConfig = SchedulerConfig()
    replay_buffer: ReplayBufferConfig = ReplayBufferConfig()
    agent: AgentConfig = AgentConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return cls.model_validate(config)


parser = ArgumentParser(prog="DQN", description="Train DQN model")
parser.add_argument("--config", "-c", dest="config", type=Path, help="Path to the training configuration file.")
parser.add_argument("--env", "-e", dest="env", type=str, default=None, help="Gymnasium environment")
parser.add_argument("--device", "-d", dest="device", type=str, default="cpu", help="Torch device to use.")


def main(
    config: TrainConfig,
    env: gym.Env,
    device: str,
):
    device = torch.device(device)
    scheduler = LinearEpsilonScheduler(train_config.scheduler)
    model = Model(nb_valid_actions=env.action_space.n, epsilon_scheduler=scheduler)
    model.to(device)
    # TODO make scheduler optional or maybe copy first model
    target_model = Model(
        nb_valid_actions=env.action_space.n, epsilon_scheduler=scheduler
    )
    target_model.to(device)
    replay_buffer = ReplayBuffer(
        config.replay_buffer, frame_shape=env.observation_space.shape[1:]
    )
    agent = Agent(
        config=config.agent,
        env=env,
        replay_buffer=replay_buffer,
        model=model,
        target_model=target_model,
        device=device,
    )
    agent.warmup()
    agent.train()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    train_config = TrainConfig.from_yaml(args["config"])
    env_name = args["env"] or "ALE/Breakout-v5"
    env = gym.make(env_name, frameskip=4, render_mode=None)
    env.close()
    env = GrayscaleObservation(env, keep_dim=False)
    # NOTE original paper mentions reshaping to (100,84) and then cropping to (84, 84)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStackObservation(env, stack_size=4)
    main(config=train_config, env=env, device=args["device"])
