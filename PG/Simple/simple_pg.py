#############################################################
# This file implement simple policy gradient algorithm
# The algorithm demonstration in simple_policy_gradient.md
# Simple policy gradient is only for 1-dimentional state
# For simply, no discount for reward
#############################################################



import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.optim import RMSprop

import Environment.Env as Env




	
class PGAgent(nn.Module):
	def __init__(self, observation_size, action_size, start_action):
		super(PGAgent, self).__init__()
		self.observation_size = observation_size
		self.action_size = action_size
		self.start_action = start_action

		# Simple network for simple problem
		self.layers = nn.Sequential(
			nn.Linear(observation_size, 24),
			nn.ReLU(),
			nn.Linear(24, 48),
			nn.ReLU(),
			nn.Linear(48, action_size)
		)

	def forward(self, x):
		return self.layers(x)

	def act(self, state):
		return Categorical(logits=self.forward(state).squeeze(0)).sample().item()

	def log_likelihood(self, state_batch, action_batch):
		return Categorical(logits=self.forward(state_batch)).log_prob(action_batch)

	def save(self, name='PGAgent.pth'):
		torch.save(self.state_dict(), 'trained_model/' + name)


class TrainAgent():
	def __init__(self, args):
		self.device = args.device
		self.env = Env.Wrapped_Env(args)
		self.agent = PGAgent(
			self.env.observation_space.shape[0],
			self.env.action_space.n,
			args.start_action).to(self.device)
		self.batch_size = 5
		self.eposide_i = 0
		self.step = 0
		self.eposide_training = args.eposide_training
		self.optimizer = optim.RMSprop(
			self.agent.parameters(), lr=1e-3)
		self.pretrained_agent = args.pretrained_agent
		# Prepare LogFile
		self._log(
			'eposide,reward,step\n', 
			'train_eposide_reward.txt',
			False)

	def _log(self, info, filename='train_eposide_reward.txt', is_append=True):
		if is_append:
			with open('Log/' + filename, 'a') as logf:
				logf.write(info)
		else:
			with open('Log/' + filename, 'w') as logf:
				logf.write(info)

	def train(self):
		for i in range(self.eposide_training):
			self._train_one_epoch()
			print(f'train epoch ---->> {i}')
		self.agent.save(self.pretrained_agent)

	def _train_one_epoch(self):
		# Count for eposides
		eposide_n = 0
		# Initialize 
		obs, done, eposide_reward = self.env.reset(), False, 0
		# Store the trajectory
		record = {'state':[], 'action':[], 'eposide_reward':[]}

		while True:
			state = torch.tensor(
				obs, dtype=torch.float32).to(self.device)
			action = self.agent.act(state.unsqueeze(0))
			record['state'].append(state)
			record['action'].append(action)
			obs, reward, done, info = self.env.step(action)
			self.step += 1
			# Don't use the wrappered information
			eposide_reward += reward

			if done:
				# make cumulative reward (weight) to be smaller for optimization.
				record['eposide_reward'] += [eposide_reward] * (
					len(record['state']) - len(record['eposide_reward']))
				eposide_n += 1
				self.eposide_i += 1
				self._log(f'{self.eposide_i},{eposide_reward},{self.step}\n')

				if eposide_n >= self.batch_size:
					self._update_parameter(record, eposide_n)
					break
				else:
					obs, done, eposide_reward = self.env.reset(), False, 0

	def _update_parameter(self, record, eposide_n):
		states = torch.stack(record['state'])
		actions = torch.tensor(record['action']).unsqueeze(0
			).float().to(self.device)
		weights = torch.tensor(record['eposide_reward']).unsqueeze(0
			).float().to(self.device)
		logprob = self.agent.log_likelihood(states, actions)
		logprob_weight_average = -(logprob * weights).sum() / eposide_n
		self.optimizer.zero_grad()
		logprob_weight_average.backward()
		self.optimizer.step()


class EvaluateAgent():
	def __init__(self, args):
		self.device = args.device
		self.path = 'trained_model/' + args.pretrained_agent
		self.env = Env.Wrapped_Env(args)
		self.agent = PGAgent(
			self.env.observation_space.shape[0],
			self.env.action_space.n,
			args.start_action).to(self.device)
		self.start_action = args.start_action

	def evaluate(self):
		self.agent.load_state_dict(
			torch.load(self.path, map_location=self.device))

		# begin to evaluate
		obs, done, eposide_reward = self.env.reset(), False, 0
		while not done:
			state = torch.tensor(obs).float().unsqueeze(0
				).to(self.device)
			action = self.agent.act(state)
			obs, reward, done, info = self.env.step(action)
			eposide_reward += reward
			self.env.render()

		print(f'reward---->>{eposide_reward}')
