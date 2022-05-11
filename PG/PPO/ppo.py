#############################################################
# This file implement Proximal Policy Optimization Algorithm.
# Clipping objective function is used.
# The algorithm demonstration in my blog.
# only for 1-dimentional state.
# For simply, only finite-horizon undiscounted is considered.
#############################################################


import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.optim import RMSprop

import Environment.Env as Env


class Utility():
	"""
	Comprises common functions.
	"""
	def __init__(self):
		pass

	def cumulate_from_now(self, rewards):
		# cumulate the reward to go
		# rewards--->> numpy array
		return np.cumsum(rewards[::-1])[::-1]
		
	def td_error(self, rewards, values):
		deltas = values[1:] + rewards[:-1] - values[:-1]
		return np.append(deltas, rewards[-1] - values[-1])

	def advantage(self, deltas, lam):
		power_num = np.array(range(len(deltas)))
		discounts = lam ** power_num
		deltas_dc = deltas * discounts
		cum_deltas_dc = self.cumulate_from_now(deltas_dc)
		return cum_deltas_dc / discounts


class Experience():
	'''
	Experience store the trajectors from agent interacting with Environment.
	Comprises two parts: trace store the trajectory with one eposide,
	traces store trajectorys.
	'''
	def __init__(self):
		self.trace = []
		self.traces = []

	def push(self, obs, action, reward):
		self.trace.append((obs, action, reward))

	def finish(self):
		self.traces.append(self.trace.copy())
		self.trace.clear()

	def clear(self):
		self.trace.clear()
		self.traces.clear()

	def get(self):
		return self.traces


class ValueApproximator(nn.Module):
	'''
	Approximate value of policy at state s.
	Using for advantage estimator.
	The core is a MLP network.
	'''
	def __init__(self, observation_size):
		super().__init__()
		# Simple network for simple problem
		self.layers = nn.Sequential(
			nn.Linear(observation_size, 24),
			nn.ReLU(),
			nn.Linear(24, 48),
			nn.ReLU(),
			nn.Linear(48, 1)
		)

	def forward(self, x):
		return self.layers(x)


class Policy(nn.Module):
	'''
	Policy pi to react to obs.
	The core is a MLP network.	
	'''
	def __init__(self, observation_size, action_size):
		super().__init__()
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
		

class PPOAgent():
	'''
	Comprise value approximator and policy.
	'''
	def __init__(self, observation_size, action_size, device):
		self.value_function = ValueApproximator(observation_size).to(device)
		self.policy = Policy(observation_size, action_size).to(device)

	def save(self, v_name='value_function.pth', p_name='policy.pth'):
		torch.save(
			self.value_function.state_dict(), 'trained_model/' + v_name)
		torch.save(
			self.policy.state_dict(), 'trained_model/' + p_name)


class TrainAgent():
	'''
	Train the agent with experience interacting
	with environment.
	'''
	def __init__(self, args):
		self.device = args.device
		self.env = Env.Wrapped_Env(args)
		self.agent = PPOAgent(
			self.env.observation_space.shape[0],
			self.env.action_space.n,
			self.device)
		self.exp = Experience()
		self.utility = Utility()
		# hyperparameter
		self.lam = 0.95
		self.epsilon = 0.2
		self.batch_trajectory = 5
		self.eposide_i = 0
		self.step = 0
		self.epochs = args.eposide_training
		self.policy_optimize_iterate_epoch = 80
		self.value_optimize_iterate_epoch = 80
		self.kl_constrain = 0.02
		# store last model to compare
		self.policy_last = Policy(
			self.env.observation_space.shape[0],
			self.env.action_space.n).to(args.device)
		self.policy_last.load_state_dict(self.agent.policy.state_dict())

		# optimizer
		self.value_optimizer = optim.RMSprop(
			self.agent.value_function.parameters(), lr=1e-3)
		self.policy_optimizer = optim.RMSprop(
			self.agent.policy.parameters(), lr=1e-3)
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
		for i in range(self.epochs):
			self._train_one_epoch()
			print(f'train epoch ---->> {i}')
			if i % 100 == 0:
				self.agent.save()

	def _train_one_epoch(self):
		self.exp.clear()
		for i in range(self.batch_trajectory):
			obs, done, eposide_reward = self.env.reset(), False, 0

			while not done:
				state_t = torch.tensor(
					obs, dtype=torch.float32).unsqueeze(0).to(self.device)
				action = self.agent.policy.act(state_t)
				next_obs, reward, done, info = self.env.step(action)
				self.step += 1
				# Don't use the wrappered information
				eposide_reward += reward
				# Push into experience
				self.exp.push(obs, action, reward)
				# swap obs
				obs = next_obs

				if done:
					# finish one trajectory
					self.exp.finish()
					self.eposide_i += 1
					self._log(f'{self.eposide_i},{eposide_reward},{self.step}\n')

		# update the parameter of value_function and policy
		self._update()

	def _update(self):
		for i in range(self.policy_optimize_iterate_epoch):
			self._update_policy()
		self.policy_last.load_state_dict(self.agent.policy.state_dict())

		for i in range(self.value_optimize_iterate_epoch):
			self._update_value_function()

	def _target_policy(self, states, actions, advs):
		states_t = torch.tensor(states).float().to(self.device)
		actions = torch.tensor(actions).float().unsqueeze(0).to(self.device)
		# (n)
		advs_t = torch.tensor(advs).float().to(self.device)
		# (n)	
		logprob = self.agent.policy.log_likelihood(
			states_t, actions).squeeze(0)
		logprob_last= self.policy_last.log_likelihood(
			states_t, actions).detach().squeeze(0)

		ratio = torch.exp(logprob - logprob_last)
		target = -torch.minimum(
			ratio * advs_t, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advs_t).mean()

		return target

	def _target_value(self, states, rewards_go):
		states_t = torch.tensor(states).float().to(self.device)
		values_t = self.agent.value_function(states_t).squeeze()
		rewards_go_t = torch.tensor(rewards_go.copy()).float().to(self.device)
		target = ((values_t - rewards_go_t) ** 2).mean() 
		return target

	def _update_policy(self):
		policy_target = 0
		value_target = 0

		for tr in self.exp.get():
			obses, actions, rewards = zip(*tr)
			states = np.array(obses)
			actions = np.array(actions)
			rewards = np.array(rewards)
			values = self.agent.value_function(
				torch.tensor(states).float().to(self.device)
				).squeeze().detach().numpy()
			deltas = self.utility.td_error(rewards, values)
			advs = self.utility.advantage(deltas, self.lam)
			policy_target += self._target_policy(states, actions, advs)

		policy_target /= self.batch_trajectory

		self.policy_optimizer.zero_grad()
		policy_target.backward()
		self.policy_optimizer.step()

	def _update_value_function(self):
		value_target = 0
		for tr in self.exp.get():
			obses, actions, rewards = zip(*tr)
			states = np.array(obses)
			rewards = np.array(rewards)
			rewards_go = self.utility.cumulate_from_now(rewards)
			value_target += self._target_value(states, rewards_go)
		value_target /= self.batch_trajectory

		self.value_optimizer.zero_grad()
		value_target.backward()
		self.value_optimizer.step()


class EvaluateAgent():
	def __init__(self, args):
		self.device = args.device
		self.env = Env.Wrapped_Env(args)
		self.agent = PPOAgent(
			self.env.observation_space.shape[0],
			self.env.action_space.n,
			self.device)

	def evaluate(self):
		self.agent.value_function.load_state_dict(
			torch.load('trained_model/value_function.pth', map_location=self.device))
		self.agent.policy.load_state_dict(
			torch.load('trained_model/policy.pth', map_location=self.device))

		# begin to evaluate
		obs, done, eposide_reward = self.env.reset(), False, 0
		while not done:
			state_t = torch.tensor(obs).float().unsqueeze(0
				).to(self.device)
			action = self.agent.policy.act(state_t)
			obs, reward, done, info = self.env.step(action)
			eposide_reward += reward
			self.env.render()

		print(f'reward---->>{eposide_reward}')
