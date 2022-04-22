import os
import argparse
import torch
import gym
import DQN.dqn as dqn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse():
    parser = argparse.ArgumentParser(description='DQN Arguments')

    parser.add_argument('--train', default=1, type=int) # 1:train, 0:evaluate
    parser.add_argument('--env_name', default='ALE/Pong-v5', help='environment name')
    parser.add_argument('--exp_capacity', default=200000, type=int)
    parser.add_argument('--device', default=device)
    ## epsilon decay parameter
    parser.add_argument('--episode_training', default=100000, type=int)
    parser.add_argument('--epsilon_start', default=50000, type=int)
    parser.add_argument('--epsilon_end', default=1000000, type=int)
    ## stack frames
    parser.add_argument('--stack_frame', default=4, type=int)
    ## update frequency for agent & target agent
    parser.add_argument('--update_target_agent', default=10000, type=int)
    parser.add_argument('--update_agent', default=4, type=int)
    ## pretained agent to be loaded to evaluate
    parser.add_argument('--pretrained_agent', default='target_agent_weight.pth')

    args = parser.parse_args()
    return args

def run(args):
    if args.train == 1:
        train_agent = dqn.TrainAgent(args)
        train_agent.train()
    else:
        evaluate_agent = dqn.EvaluateAgent(args)
        evaluate_agent.evaluate()


def main():
    args = parse()
    run(args)

if __name__ == '__main__':
    main()
