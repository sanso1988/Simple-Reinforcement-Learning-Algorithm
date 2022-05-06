import gym
import math

################################################################
# wrap the environment
# put all customize information include ('reward', 'done') in info
# customize information is for training
################################################################

class Wrapped_Env():
	"""wrap the gym environment to fit the training"""
	def __init__(self, args, render=False):
		self.env_name = args.env_name
		self.env = gym.make(self.env_name, render_mode='human') if render else gym.make(self.env_name)
		self.lost_life_done = args.lost_life_done
		self.lives = 0
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space

	def reset(self):
		obs = self.env.reset()
		return obs

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		info['done'] = done
		info['reward'] = self._clip(reward)
		
		if self.lost_life_done == 1 and info.get('lives') is not None:
			if info['lives'] < self.lives:
				info['done'] = True
			self.lives = info['lives']
		return obs, reward, done, info

	def render(self):
		self.env.render()
		
	def _clip(self, reward):
		if reward > 1:
			reward = 1
		elif reward < -1:
			reward = -1
		return reward