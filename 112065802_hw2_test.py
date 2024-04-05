import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import torch
import torch.nn as nn

class Agent:
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.model = torch.load('112065802_hw2_data')

    def act(self, observation):
        return torch.argmax(self.model.act(observation)).item()
        # return self.action_space.sample()

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

obs = env.reset()
agent = Agent()
reward = 0
done = False

while not done:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()

print("score:", total_reward)