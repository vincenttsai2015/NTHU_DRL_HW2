import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T

import random, time, datetime, os, itertools
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import namedtuple, deque
from PIL import Image

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# Environment initialization
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# done = True
# for step in range(5000):
#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#     # env.render()
# env.close()

# Number of states
N_STATE = env.observation_space.shape[0]
print(f'Number of states: {N_STATE}')
# Number of actions
N_ACTION = env.action_space.n
print(f'Number of actions: {N_ACTION}')
ACT_SPACE = env.get_action_meanings()
print(f'Action space: {ACT_SPACE}')

BUFFER_SIZE = int(5e3)  # size of replay buffer
BATCH_SIZE = 128         # mini-batch size
GAMMA = 0.95
TAU = 0.001
LR = 0.0005
EPSILON = 0.05
UPDATE_EVERY = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, priority=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer (chosen as multiple of num agents)
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.states = torch.zeros((buffer_size,)+state_size).to(device)
        self.next_states = torch.zeros((buffer_size,)+state_size).to(device)
        self.actions = torch.zeros(buffer_size,1, dtype=torch.long).to(device)
        self.rewards = torch.zeros(buffer_size, 1, dtype=torch.float).to(device)
        self.dones = torch.zeros(buffer_size, 1, dtype=torch.float).to(device)
        self.e = np.zeros((buffer_size, 1), dtype=np.float64)
        
        self.priority = priority

        self.ptr = 0
        self.n = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.states[self.ptr] = torch.from_numpy(state).to(device)
        self.next_states[self.ptr] = torch.from_numpy(next_state).to(device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.ptr = 0
            self.n = self.buffer_size

    def sample(self, get_all=False):
        """Randomly sample a batch of experiences from memory."""
        n = len(self)
        if get_all:
            return self.states[:n], self.actions[:n], self.rewards[:n], self.next_states[:n], self.dones[:n]
        # else:
        if self.priority:
            idx = np.random.choice(n, self.batch_size, replace=False, p=self.e)
        else:
            idx = np.random.choice(n, self.batch_size, replace=False)
        
        states = self.states[idx]
        next_states = self.next_states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]
        
        return (states, actions, rewards, next_states, dones), idx
      
    def update_error(self, e, idx=None):
        e = torch.abs(e.detach())
        e = e / e.sum()
        if idx is not None:
            self.e[idx] = e.cpu().numpy()
        else:
            self.e[:len(self)] = e.cpu().numpy()
        
    def __len__(self):
        if self.n == 0:
            return self.ptr
        else:
            return self.n

class DuelingDQN(nn.Module):
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

# Agent
class Agent:
    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE

        # Q-Network Architecture
        self.qnetwork_local = DuelingDQN(self.state_size[0], self.action_size, 30).to(device)
        self.qnetwork_target = DuelingDQN(self.state_size[0], self.action_size, 30).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(self.state_size, (self.action_size,), BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences, idx = self.memory.sample()
                e = self.learn(experiences)
                self.memory.update_error(e, idx)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > EPSILON:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update_error(self):
        states, actions, rewards, next_states, dones = self.memory.sample(get_all=True)
        with torch.no_grad():
            # dueling DQN
            old_val = self.qnetwork_local(states).gather(-1, actions)
            actions = self.qnetwork_local(next_states).argmax(-1, keepdim=True)
            maxQ = self.qnetwork_target(next_states).gather(-1, actions)
            target = rewards+GAMMA*maxQ*(1-dones)
            # Normal DQN
            # maxQ = self.qnetwork_target(next_states).max(-1, keepdim=True)[0]
            # target = rewards+GAMMA*maxQ*(1-dones)
            # old_val = self.qnetwork_local(states).gather(-1, actions)
            e = old_val - target
            self.memory.update_error(e)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        ## compute and minimize the loss
        self.optimizer.zero_grad()
        # Dueling DQN
        old_val = self.qnetwork_local(states).gather(-1, actions)
        with torch.no_grad():
            next_actions = self.qnetwork_local(next_states).argmax(-1, keepdim=True)
            maxQ = self.qnetwork_target(next_states).gather(-1, next_actions)
            target = rewards+GAMMA*maxQ*(1-dones)
        # Normal DQN
        # with torch.no_grad():
        #     maxQ = self.qnetwork_target(next_states).max(-1, keepdim=True)[0]
        #     target = rewards+GAMMA*maxQ*(1-dones)
        # old_val = self.qnetwork_local(states).gather(-1, actions)   
        
        loss = F.mse_loss(old_val, target)
        loss.backward()
        self.optimizer.step()

        # update target network 
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 
        
        return old_val - target

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model(self):
        torch.save(self.qnetwork_local.state_dict(), '112065802_hw2_data')

def preprocess(frame):
    frame = frame.sum(axis=-1)/765
    frame = frame[20:210,:]
    frame = frame[::2,::2]
    return frame

# Training
episode = 10000
discount_rate = .99
noise = 0.05
noise_decay = 0.99
tmax = 500

# keep track of progress
sum_rewards = []

# keep track of frames
FRAME_SHAPE = (95, 128)
MAX_FRAMES = 4
nn_frames = deque(maxlen=MAX_FRAMES)
for i in range(MAX_FRAMES):
    nn_frames.append(np.zeros(FRAME_SHAPE))

ACTION_SIZE = 12 #len(valid_actions)
STATE_SIZE = (MAX_FRAMES,) + FRAME_SHAPE
print(f'Size of states: {STATE_SIZE}')
agent = Agent()

for e in range(episode):
    obs = env.reset()
    # print(f'Size of currently observed state: {obs.shape}')
    sum_reward = 0
    
    for i in range(MAX_FRAMES):
        nn_frames.append(np.zeros(FRAME_SHAPE))
    # print(f'Size of preprocessed observed state: {preprocess(obs).shape}')
    nn_frames.append(np.copy(preprocess(obs)))
    states = np.array(nn_frames)
    # print(f'Size of frame states: {states.shape}')
    for t in range(tmax):
        actions = agent.act(states)
        obs, reward, done, _ = env.step(actions)
        nn_frames.append(np.copy(preprocess(obs)))
        next_states = np.array(nn_frames)
        
        agent.step(states, int(actions), int(reward), next_states, int(done))
        sum_reward += reward
        states = next_states

        if done or reward < -10:
            break
    
    agent.update_error()
    # get the average reward of the parallel environments
    sum_rewards.append(sum_reward)
    noise = noise * noise_decay
    
    print('\rEpisode {}\tCurrent Score: {:.2f}'.format(e, sum_rewards[-1]), end="")
    # display some progress every 20 iterations
    if (e+1) % (episode // 20) ==0:
        print(" | Episode: {0:d}, average score: {1:f}".format(e+1,np.mean(sum_rewards[-20:])))
        agent.save_model()