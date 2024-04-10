import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random, time, datetime, os
import cv2
from pathlib import Path
from collections import deque

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# Dueling Double DQN Architecture
class D3QN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        
        self.conv1 = nn.ReLU(nn.Conv2d(c, 32, kernel_size=8, stride=4))
        self.conv2 = nn.ReLU(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = nn.ReLU(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28224, 512)
        
        self.advantage = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        self.value = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value, advantage

# Agent
class Agent:
    def __init__(self):     
        # for local test
        self.use_cuda = torch.cuda.is_available()
        self.net = D3QN((4, 84, 84), 12)
        if self.use_cuda:
            self.net.load_state_dict(torch.load('112065802_hw2_data')).cpu()
        else:
            self.net.load_state_dict(torch.load('112065802_hw2_data'))
        self.frames = deque(maxlen=4)
        self.curr_step = 0
        self.memory = deque(maxlen=100000)

    def act(self, observation):
        preprocess_obs = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        preprocess_obs = cv2.resize(preprocess_obs, (84, 84), interpolation=cv2.INTER_AREA)
        while len(self.frames) < 3:
            self.frames.append(preprocess_obs)
        self.frames.append(preprocess_obs)
        preprocess_obs = torch.from_numpy(np.array(self.frames) / 255).float().unsqueeze(0)
        _, action_values = self.net(preprocess_obs)
        action_idx = torch.argmax(action_values, axis=1).item()
        
        # increment step
        self.curr_step += 1
        
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Add the experience to memory"""
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.FloatTensor(state.copy())
        next_state = torch.FloatTensor(next_state.copy())
        action = torch.LongTensor([action])
        reward = torch.DoubleTensor([reward])
        done = torch.BoolTensor([done])

        self.memory.append((state, next_state, action, reward, done))


if __name__=='__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    env.reset()
    mario = Agent()

    total_reward = 0
    episodes = 50

    for e in range(episodes):
        state = env.reset()
        episode_reward = 0
        print(f'Episode {e}')
        while True:
            # env.render()
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)
            mario.cache(state, next_state, action, reward, done)
            episode_reward += reward
            
            state = next_state

            if done or info['flag_get']:
                break
        
        print(f'Episode reward in episode {e}: {episode_reward}')
        total_reward += episode_reward

    avg_reward = total_reward/50
    print(f'Average reward: {avg_reward}')