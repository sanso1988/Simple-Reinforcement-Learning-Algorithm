import argparse
import torch
import gym
import DQN.dqn as dqn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse():
    parser = argparse.ArgumentParser(description='DQN Arguments')
    parser.add_argument('--env_name', default='ALE/Pong-v5', help='environment name')
    parser.add_argument('--exp_capacity', default=200000, type=int)
    parser.add_argument('--episode_training', default=1000000, type=int)
    parser.add_argument('--epsilon_start', default=50000, type=int)
    parser.add_argument('--epsilon_end', default=1000000, type=int)
    parser.add_argument('--stack_frame', default=4, type=int)
    parser.add_argument('--update_target_agent', default=10000, type=int)
    parser.add_argument('--update_agent', default=1, type=int)
    parser.add_argument('--learn_start', default=50000, type=int)
    args = parser.parse_args()
    return args

def run(args):
	env = gym.make(args.env_name)
	exp = dqn.ExpReplay(capacity=args.exp_capacity)
	agent = dqn.DQNAgent(env.action_space.n).to(device)
	target_agent = dqn.DQNAgent(env.action_space.n).to(device)
	train_agent = dqn.TrainAgent(env, agent, target_agent, exp, device, args)
	train_agent.train()

def main():
    args = parse()
    run(args)

if __name__ == '__main__':
    main()
