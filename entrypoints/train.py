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
import torch

NUM_ENVS = 16

class EnvConfig(BaseModel):
    env_name: str = "ALE/Breakout-v5"
    frameskip: int = 4
    render_mode: str = None


class TrainConfig(BaseModel):
    scheduler: SchedulerConfig = SchedulerConfig()
    replay_buffer: ReplayBufferConfig = ReplayBufferConfig()
    agent: AgentConfig = AgentConfig()
    env: EnvConfig = EnvConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return cls.model_validate(config)


parser = ArgumentParser(prog="DQN", description="Train DQN model")
parser.add_argument(
    "--config",
    "-c",
    dest="config",
    type=Path,
    help="Path to the training configuration file.",
)
parser.add_argument(
    "--device",
    "-d",
    dest="device",
    type=str,
    default="cpu",
    help="Torch device to use.",
)
parser.add_argument(
    "--path",
    "-p",
    dest="saved_model_path",
    type=Path,
    default="model.pt",
    help="Path where the trained model will be saved.",
)

def make_env(env_config: EnvConfig) -> gym.Env:
    def _init():
        env = gym.make(env_config.env_name, frameskip=env_config.frameskip, render_mode=env_config.render_mode)
        env = GrayscaleObservation(env, keep_dim=False)
        # NOTE original paper mentions reshaping to (100,84) and then cropping to (84, 84)
        env = ResizeObservation(env, shape=(84, 84))
        env = FrameStackObservation(env, stack_size=4)
        return env
    return _init

def main(config: TrainConfig, env: gym.Env, device: str, saved_model_path: str):
    device = torch.device(device)
    scheduler = LinearEpsilonScheduler(train_config.scheduler)
    action_space = env.action_space[0].n
    model = Model(nb_valid_actions=action_space, epsilon_scheduler=scheduler)
    model.to(device)
    target_model = Model(
        nb_valid_actions=action_space, epsilon_scheduler=scheduler
    )
    # target_model.to(device)
    replay_buffer = ReplayBuffer(
        config.replay_buffer, frame_shape=env.observation_space.shape[2:]
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
    torch.save(agent.model.state_dict(), saved_model_path)


if __name__ == "__main__":
    gym.register_envs(ale_py)
    args = vars(parser.parse_args())
    train_config = TrainConfig.from_yaml(args["config"])
    env = gym.vector.AsyncVectorEnv([make_env(train_config.env) for _ in range(NUM_ENVS)])
    main(config=train_config, env=env, device=args["device"], saved_model_path=args["saved_model_path"])
