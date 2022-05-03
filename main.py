import os
import argparse
import torch
import gym
import DQN.dqn as dqn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse():
    parser = argparse.ArgumentParser(description='DQN Arguments')

    # evironment settting
    parser.add_argument('--env_name', default='BreakoutDeterministic-v4', help='environment name')
    parser.add_argument('--start_action', default=1, type=int, help='action to start game')
    parser.add_argument('--lost_life_done', default=1, type=int, help='lost life sets termial state')
    # algorithm & train
    parser.add_argument('--algorithm', default='DQN', help='choose algorithm to train')
    parser.add_argument('--doubleQ', default=1, type=int, help='using double q algorithm only valid when algorithm is DQN')
    parser.add_argument('--train', default=1, type=int, help='train or evaluate') # 1:train, 0:evaluate
    parser.add_argument('--exp_capacity', default=200000, type=int, help='experience memory size')
    parser.add_argument('--device', default=device)
    ## epsilon decay parameter
    parser.add_argument('--episode_training', default=10000000, type=int)
    parser.add_argument('--epsilon_start', default=50000, type=int)
    parser.add_argument('--epsilon_end', default=1000000, type=int)
    ## stack frames
    parser.add_argument('--stack_frame', default=4, type=int)
    ## update frequency for agent & target agent
    parser.add_argument('--update_target_agent', default=30000, type=int)
    parser.add_argument('--update_agent', default=4, type=int)
    ## pretained agent to be loaded to evaluate or trained
    parser.add_argument('--load_pretrained_agent', default=0, type=int)
    parser.add_argument('--pretrained_agent', default='target_agent_weight.pth')

    args = parser.parse_args()
    return args

def run(args):
    # set algorithm, only DQN avaliable now
    if args.algorithm == 'DQN':
        algorithm = dqn
    else:
        os._exit(0)

    if args.train == 1:
        train_agent = algorithm.TrainAgent(args)
        train_agent.train()
    else:
        evaluate_agent = algorithm.EvaluateAgent(args)
        evaluate_agent.evaluate()

def main():
    args = parse()
    run(args)

if __name__ == '__main__':
    main()
