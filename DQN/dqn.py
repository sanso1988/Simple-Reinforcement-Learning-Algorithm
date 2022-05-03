import os
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

import gym
import collections
import Environment.Env as Env

class Utility():
	def __init__(self, args):
		self.stack_frame = args.stack_frame
		self.device = args.device

	def _transform_frame(self, frame):
		mini_frame = cv2.cvtColor(cv2.resize(frame, (84,84)), 
			cv2.COLOR_BGR2GRAY)
		return mini_frame

	def _update_state(self, frame, state):
		# 84 * 84
		mini_frame = self._transform_frame(frame)
		mini_frame_tensor = torch.tensor(
			mini_frame, dtype=torch.uint8).to(self.device)

		if state.numel() == 0:
			# 1 * 1 * 84 * 84
			state = mini_frame_tensor.unsqueeze(0).unsqueeze(0)
		elif len(state[0]) < self.stack_frame:
			state = torch.cat((state.squeeze(0), mini_frame_tensor.unsqueeze(0))).unsqueeze(0)
		else:
			state = torch.cat(
				(state.squeeze(0)[1:,:,:], mini_frame_tensor.unsqueeze(0))).unsqueeze(0)

		return state

class ExpReplay():
	def __init__(self, capacity):
		self.memory = collections.deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def sample(self, batch_size=32):
		memory_sample = random.sample(self.memory, batch_size)
		states, actions, rewards, next_states, dones = zip(*memory_sample)
		return states, actions, rewards, next_states, dones

	def __len__(self):
		return len(self.memory)

class DQNAgent(nn.Module):
	def __init__(self, action_size, stack_frame, start_action):
		super(DQNAgent, self).__init__()
		self.action_size = action_size
		self.start_action = start_action

		self.layers = nn.Sequential(
			nn.Conv2d(stack_frame, 32, 8, stride=4),
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

	def act(self, state, epsilon, evaluate=False):
		if random.random() > epsilon:
			predq = self.layers(state)
			action = predq[0].argmax().item()
		else:
			action = random.randint(0, self.action_size - 1) if not evaluate else self.start_action
		return action

class TrainAgent():
	def __init__(self, args):
		self.device = args.device
		self.env = Env.Wrapped_Env(args)
		self.agent = DQNAgent(self.env.action_space.n, args.stack_frame, args.start_action
			).to(self.device)
		self.target_agent = DQNAgent(self.env.action_space.n, args.stack_frame, args.start_action
			).to(self.device)
		self.exp = ExpReplay(args.exp_capacity)
		self.doubleQ = args.doubleQ
		self.start_action = args.start_action
		# Training Parameters
		self.episode_training = args.episode_training
		self.gamma = 0.99
		self.batch_size = 32
		self.stack_frame = args.stack_frame
		self.epsilon = 1
		self.epsilon_start = args.epsilon_start
		self.epsilon_end = args.epsilon_end
		self.epsilon_min = 0.1
		self.update_agent = args.update_agent
		self.update_target_agent = args.update_target_agent
		# Loss & Optimize
		self.optimizer = optim.RMSprop(
			self.agent.parameters(), lr=0.00001, alpha=0.95, momentum=0.95)
		self.loss = nn.HuberLoss()
		# Record Steps
		self.steps = 0
		# Utility Functions
		self.utility = Utility(args)
		# Path to save model
		self.model_path = 'trained_model/' + args.pretrained_agent
		# Prepare LogFile
		self._log('eposide,reward,step\n', 'train_eposide_reward.txt', False)
		# pretained model path
		self.path = 'trained_model/' + args.pretrained_agent
		if args.load_pretrained_agent == 1:
			self._load_agent()

	def _load_agent(self):
		print('pretrained model load...')
		self.agent.load_state_dict(torch.load(self.path, map_location=self.device))
		self.target_agent.load_state_dict(torch.load(self.path, map_location=self.device))

	def _log(self, info, filename='train_eposide_reward.txt', is_append=True):
		if is_append:
			with open('Log/' + filename, 'a') as logf:
				logf.write(info)
		else:
			with open('Log/' + filename, 'w') as logf:
				logf.write(info)

	def _epsilon_decay(self):
		self.epsilon = 1 - (self.steps / self.epsilon_end) * 0.9
		self.epsilon = max(self.epsilon, self.epsilon_min)
		
	def train(self):
		for episode_i in range(self.episode_training):
			obs, done, state, episode_reward = self.env.reset(), False, torch.tensor([]).to(self.device), 0

			while not done:
				state = self.utility._update_state(obs, state)

				# if frames is not enough to make decision
				if len(state[0]) < self.stack_frame:
					obs, reward, done, info = self.env.step(self.start_action)
				else:
					# interaction with environment. state normalization is very important
					if info['done']:
						action = self.start_action
					else:
						action = self.agent.act(state / 255.0, self.epsilon)
					obs, reward, done, info = self.env.step(action)

					# store experience
					self.exp.push(state, action, info['reward'], self.utility._update_state(obs, state), info['done'])
					self.steps += 1
					episode_reward += reward

					# epsilon decay
					if self.steps >= self.epsilon_start:
						self._epsilon_decay()

					# learn from experience
					if self.steps % self.update_agent == 0:
						# don't learn before epsilon decay
						if len(self.exp) >= self.epsilon_start:
							self._train_one_epoch()

					# store target agent & save target agent weight
					if self.steps % self.update_target_agent == 0:
						self.target_agent.load_state_dict(self.agent.state_dict())
						if self.steps % (self.update_target_agent * 10) == 0:
							torch.save(self.target_agent.state_dict(), self.model_path)

			self._log(f'{episode_i},{episode_reward},{self.steps}\n')

	def _train_one_epoch(self):
		# state, next_state is tensor, action, reward is scalar tuple
		states, actions, rewards, next_states, dones = self.exp.sample(self.batch_size)
		# state normalization is very important for proper gradients
		states = torch.cat(states).float() / 255.0
		next_states = torch.cat(next_states).float() / 255.0
		# calculate target_q = reward + discount * q_star
		if self.doubleQ == 0:
			target_q = (
				torch.tensor(rewards, dtype=torch.float32).to(self.device) + 
				self.gamma * self.target_agent(next_states).max(1)[0] * 
				(1 - torch.tensor(dones, dtype=torch.float32)).to(self.device)
				)
		else:
			target_q = (
				torch.tensor(rewards, dtype=torch.float32).to(self.device) + 
				self.gamma * self.target_agent(next_states).gather(1, 
				self.agent(next_states).max(1)[1].unsqueeze(1)).squeeze() *
				(1 - torch.tensor(dones, dtype=torch.float32)).to(self.device)
				)
		target_q = target_q.detach()
		# calculate q(s,a): select the action value from agent on states
		q = self.agent(states).gather(1, torch.tensor(actions).unsqueeze(1).to(self.device)).squeeze()
		self.optimizer.zero_grad()
		self.loss(q, target_q).backward()
		self.optimizer.step()
		# temp monitor
		if self.steps % 20000 == 0:
			print(q)
			print('*'*30)
			print(target_q)

class EvaluateAgent():
	"""docstring for EvaluateAgent
	Evaluate the pretrained agent performance
	"""
	def __init__(self, args):
		self.device = args.device
		self.path = 'trained_model/' + args.pretrained_agent
		self.env = Env.Wrapped_Env(args, render=True)
		self.agent = DQNAgent(self.env.action_space.n, args.stack_frame, args.start_action
			).to(self.device)
		self.utility = Utility(args)
		self.stack_frame = args.stack_frame
		self.start_action = args.start_action
		
	def evaluate(self):
		self.agent.load_state_dict(torch.load(self.path, map_location=self.device))

		# begin to evaluate
		obs, done, state = self.env.reset(), False, torch.tensor([])
		while not done:
			state = self.utility._update_state(obs, state)
			if len(state[0]) < self.stack_frame:
				obs, reward, done, _ = self.env.step(self.start_action)
			else:
				# state normalization is very important
				action = self.agent.act(state.float() / 255.0, 0.01, evaluate=True)
				obs, reward, done, info = self.env.step(action)
