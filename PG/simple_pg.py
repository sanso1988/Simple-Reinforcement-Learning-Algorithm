import torch
import torch.nn as nn
import numpy as np
import gym
import torch.optim as optim

from torch.distributions.categorical import Categorical
from torch.optim import RMSprop
from gym.wrappers import RecordVideo




	
class Agent(nn.Module):
	def __init__(self, observation_size, action_size):
		super(Agent, self).__init__()

		self.observation_size = observation_size
		self.action_size = action_size

		self.layers = nn.Sequential(
			nn.Linear(observation_size, 24),
			nn.ReLU(),
			nn.Linear(24, 48),
			nn.ReLU(),
			nn.Linear(48, action_size)
		)

	def forward(self, x):
		return self.layers(x)

	def action(self, state):
		return Categorical(logits=self.forward(state).squeeze(0)).sample().item()

	def max_likelihood(self, state_batch, action_batch):
		return Categorical(logits=self.forward(state_batch)).log_prob(action_batch)

	def save(self, name='PGAgent.pth'):
		torch.save(self, name)

class TrainAgentViaPG():
	def __init__(self, agent, env):
		self.env = env
		self.agent = agent
		self.lr = 1e-2
		self.batch_size = 5000
		self.optimizer = optim.RMSprop(self.agent.parameters())

	def train_one_epoch(self):
		record = {'state':[], 'action':[], 'reward_episode':[]}
		i_episode = 0

		state, done, reward_episode = self.env.reset(), False, 0

		while True:
			action = self.agent.action(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
			# print(action)
			record['state'].append(state)
			record['action'].append(action)
			state, reward, done, _ = self.env.step(action)
			reward_episode += reward

			if done:
				record['reward_episode'] += [reward_episode] * (len(record['state']) - len(record['reward_episode']))
				i_episode += 1

				if len(record['state']) > self.batch_size:
					logprob = self.agent.max_likelihood(torch.tensor(record['state'], dtype=torch.float32),
						torch.tensor(record['action'], dtype=torch.float32))
					logprob_weight = -(logprob * torch.tensor(record['reward_episode'], dtype=torch.float32)).mean() / i_episode
					self.optimizer.zero_grad()
					logprob_weight.backward()
					self.optimizer.step()
					break

				else:
					state, done, reward_episode = self.env.reset(), False, 0

	def train(self, times=100):
		for i in range(times):
			self.train_one_epoch()
			print(f'train epoch ---->> {i}')

class TestAgent():
	def __init__(self, agent, env):
		self.agent = agent
		self.env = env

	def test(self):
		state, done, reward_episode = self.env.reset(), False, 0

		while not done:
			action = self.agent.action(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
			state, reward, done, _ = self.env.step(action)
			reward_episode += reward
			self.env.render()

		print(f'reward---->>{reward_episode}')

if __name__ == '__main__':
	env = gym.make('CartPole-v1')
	video_env = RecordVideo(env, '')
	PGAgent = Agent(env.observation_space.shape[0], env.action_space.n)
	TrainAgent = TrainAgentViaPG(PGAgent, env)
	TestAgent = TestAgent(PGAgent, video_env)
	for i in range(2000):
		TrainAgent.train_one_epoch()
		if i % 50 == 0:
			TestAgent.test()








