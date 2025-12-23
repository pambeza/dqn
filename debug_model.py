
import gymnasium as gym
import ale_py
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation
import torch
import numpy as np
import sys
import os
from collections import defaultdict

from dqn.model import Model
from dqn.scheduler import ConstantScheduler
# Ensure we can import dqn
sys.path.append(os.getcwd())

# Register envs
gym.register_envs(ale_py)

def run_debug():
    try:
        scheduler = ConstantScheduler(0.05)
        model = Model(4, scheduler)
        model.load_state_dict(torch.load("dqn.pt", map_location=torch.device("cpu")))
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=4)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStackObservation(env, stack_size=4)

    state, _ = env.reset()
    
    print("Starting loop...")
    actions_to_count = defaultdict(int)
    # Increased steps to see more gameplay
    for i in range(100):
        # Convert state
        # Simulating what's in the notebook
        state_tensor = torch.from_numpy(np.array(state)).to(torch.float32)
        
        action = model.policy(state_tensor)
        actions_to_count[action] += 1
        print("Action: ", action)
        state, reward, done, truncated, _ = env.step(action)
        
        # If episode ends (ball lost), restart and FIRE again
        if done or truncated:
            print("Episode ended, restarting...")
            state, _ = env.reset()
            
    print(f"Actions to count: {actions_to_count}")
    env.close()

if __name__ == "__main__":
    run_debug()
