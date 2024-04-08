import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import random, time, datetime, os, itertools
import numpy as np
import time
from tqdm import tqdm
# import matplotlib.pyplot as plt
from collections import namedtuple, deque

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess(frame):
    frame = frame.sum(axis=-1)/765
    frame = frame[20:210,:]
    frame = frame[::2,::2]
    return frame

class DuelingDQN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, channels, action_size, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(channels, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        
        flat_len = 16*3*4
        self.fcval = nn.Linear(flat_len, 20)
        self.fcval2 = nn.Linear(20, 1)
        self.fcadv = nn.Linear(flat_len, 20)
        self.fcadv2 = nn.Linear(20, action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.reshape(x.shape[0], -1)
        
        advantage = F.relu(self.fcadv(x))
        advantage = self.fcadv2(advantage)
        advantage = advantage - torch.mean(advantage, dim=-1, keepdim=True)
        
        value = F.relu(self.fcval(x))
        value = self.fcval2(value)

        return value + advantage

class Agent:
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.model = DuelingDQN(4, 12, 30)
        self.model.load_state_dict(torch.load("112065802_hw2_data"), strict=False)
        # self.model = torch.load('112065802_hw2_data')

    def act(self, observation):
        nn_frames = deque(maxlen=MAX_FRAMES)
        for i in range(MAX_FRAMES):
            nn_frames.append(np.zeros(FRAME_SHAPE))
        # print(f'Size of preprocessed observed state: {preprocess(observation).shape}')
        nn_frames.append(np.copy(preprocess(observation)))
        states = np.array(nn_frames)
        
        observations = torch.from_numpy(np.copy(states)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.model(observations)

        # Epsilon-greedy action selection
        if random.random() > 0:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

# keep track of frames
FRAME_SHAPE = (95, 128)
MAX_FRAMES = 4
nn_frames = deque(maxlen=MAX_FRAMES)
    
# ACTION_SIZE = 12 #len(valid_actions)
# STATE_SIZE = (MAX_FRAMES,) + FRAME_SHAPE

agent = Agent()
tmax = 500
rs = []
xs = []
ys = []
frames = np.zeros((tmax, 240, 256, 3), dtype=np.uint8)

for e in range(50):
    obs = env.reset()
    sum_reward = 0
    for t in range(tmax):
        frames[t] = obs
        actions = agent.act(frames[t])
        obs, reward, done, info = env.step(actions)
        nn_frames.append(np.copy(preprocess(obs)))
        next_states = np.array(nn_frames)

        sum_reward += reward
        states = next_states
        rs.append(reward)
        xs.append(info['x_pos'])
        ys.append(info['y_pos'])
        if done:
            break

print('Sum of rewards is ', sum(rs))
# plt.plot(rs)
# plt.show()

# plt.plot(xs)
# plt.show()

# total_reward = 0
# total_time = 0
# time_limit = 120
# for episode in tqdm(range(50), desc="Evaluating"):
    # obs = env.reset()
    # print(f'Size of currently observed state: {obs.shape}')
    # start_time = time.time()
    # episode_reward = 0
    
    # while True:
        # action = agent.act(obs)
        # obs, reward, done, info = env.step(action)
        # episode_reward += reward

        # if time.time() - start_time > time_limit:
            # print(f"Time limit reached for episode {episode}")
            # break

        # if done:
            # break

    # end_time = time.time()
    # total_reward += episode_reward
    # total_time += (end_time - start_time)

# env.close()

# score = total_reward / 50
# print(f"Final Score: {score}")
