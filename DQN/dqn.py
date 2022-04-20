import math
import numpy as np
import random
import cv2
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import gym
from collections import deque


class ExpReplay():
	def __init__(self, capacity):
		super(ExpReplay, self).__init__()
		self.memory = deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def sample(self, batch_size=32):
		memory_sample = random.sample(self.memory, batch_size)
		states, actions, rewards, next_states, dones = zip(*memory_sample)
		return states, actions, rewards, next_states, dones

	def __len__(self):
		return len(self.memory)

class DQNAgent(nn.Module):
	def __init__(self, action_size):
		super(DQNAgent, self).__init__()
		self.action_size = action_size

		self.layers = nn.Sequential(
			nn.Conv2d(4, 32, 8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, 4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, stride=1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(3136, 512),
			nn.ReLU(),
			nn.Linear(512, self.action_size),
		)

	def forward(self, x):
		return self.layers(x)

	def act(self, state, epsilon):
		if random.random() > epsilon:
			predq = self.layers(state)
			action = predq[0].argmax().item()
		else:
			action = random.randint(0, self.action_size - 1)
		return action

class TrainAgent():
	def __init__(self, env, agent, target_agent, exp, device, args):
		self.env = env
		self.agent = agent
		self.target_agent = target_agent
		self.exp = exp
		self.device = device
		# Train Arguments		
		self.episode_training = args.episode_training
		self.gamma = 0.99
		self.batch_size = 32
		self.stack_frame = args.stack_frame
		self.epsilon = 1
		self.epsilon_start = args.epsilon_start
		self.epsilon_end = args.epsilon_end
		self.update_agent = args.update_agent
		self.update_target_agent = args.update_target_agent
		self.learn_start = args.learn_start
		self.optimizer = optim.RMSprop(
			self.agent.parameters(), lr=0.00025, alpha=0.95, eps=0.01, momentum=0.95)
		self.loss = nn.HuberLoss()
		# Record Steps
		self.steps = 0

	def _epsilon_decay(self):
		self.epsilon = 1 - self.steps / self.epsilon_end
		self.epsilon = max(self.epsilon, 0.1)
		
	def _transform_frame(self, frame):
		mini_frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
		return mini_frame

	def _update_state(self, frame, state):
		mini_frame = self._transform_frame(frame)
		mini_frame_tensor = torch.tensor(mini_frame, dtype=torch.uint8).unsqueeze(0).to(self.device)

		if len(state.squeeze(0)) == 0:
			state = mini_frame_tensor.unsqueeze(0)
		elif len(state.squeeze(0)) < self.stack_frame:
			state = torch.cat((state.squeeze(0), mini_frame_tensor)).unsqueeze(0)
		else:
			state = torch.cat((state.squeeze(0)[1:,:,:], mini_frame_tensor)).unsqueeze(0)

		return state

	def train(self):
		for episode_i in range(self.episode_training):
			obs, done, state, episode_reward = self.env.reset(), False, torch.tensor([]), 0

			while not done:
				state = self._update_state(obs, state).to(self.device)
				if len(state[0]) < self.stack_frame:
					obs, reward, done, _ = self.env.step(0)
				else:
					action = self.agent.act(state.float(), self.epsilon)
					obs, reward, done, _ = self.env.step(action)
					self.exp.push(state, action, reward, self._update_state(obs, state), done)
					self.steps += 1
					episode_reward += reward

					if self.steps % self.update_agent == 0:
						if len(self.exp) >= self.learn_start:
							self._train_one_epoch()

					if self.steps >= self.epsilon_start:
						self._epsilon_decay()

					if self.steps % self.update_target_agent == 0:
						self.target_agent.load_state_dict(self.agent.state_dict())
						if self.steps % (self.update_target_agent * 10) == 0:
							torch.save(self.target_agent, 'target_agent.pth')

			with open('eposide_log.txt', 'a') as f:
				f.write(f'{episode_i},{episode_reward},{self.steps}')
				f.write('\n')

	def _train_one_epoch(self):
		# state, next_state is tensor, action, reward is scalar
		states, actions, rewards, next_states, dones = self.exp.sample(self.batch_size)
		states = torch.cat(states).float() / 255.0
		next_states = torch.cat(next_states).float() / 255.0
		target_q = (
			torch.tensor(rewards, dtype=torch.float32).to(self.device) + 
			self.gamma * self.target_agent(next_states).max(1)[0] * 
			(1 - torch.tensor(dones, dtype=torch.float32)).to(self.device)
			)
		target_q = target_q.detach().to(self.device)
		q = self.agent(states).gather(1, torch.tensor(actions).unsqueeze(1).to(self.device)).squeeze()
		self.optimizer.zero_grad()
		self.loss(q, target_q).backward()
		self.optimizer.step()
