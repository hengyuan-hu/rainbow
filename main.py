"""Run Atari Environment with DQN."""
import argparse
import os
import random
import numpy as np
from env import Environment
import torch
from dqn import DQNAgent, PredDQNAgent
from policy import GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from model import DQNetwork, DuelingQNetwork, PredDuelingQNetwork, \
    SinglePredDuelingQNetwork
from core import ReplayMemory
from logger import Logger
from train import train, evaluate


def large_randint():
    return random.randint(int(1e5), int(1e6))


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari')
    parser.add_argument('--rom', default='roms/space_invaders.bin',
                        help='path to rom')
    parser.add_argument('--seed', default=6666999, type=int, help='Random seed')
    # parser.add_argument('--lr', default=0.00025, type=float, help='learning rate')
    # parser.add_argument('--alpha', default=0.95, type=float,
    #                     help='squared gradient momentum for RMSprop')
    # parser.add_argument('--momentum', default=0.95, type=float,
    #                     help='gradient momentum for RMSprop')
    # parser.add_argument('--rms_eps', default=0.01, type=float,
    #                     help='min squared gradient for RMS prop')
    parser.add_argument('--q_net', default='', type=str, help='load pretrained q net')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--num_iters', default=50000000, type=int)
    parser.add_argument('--replay_buffer_size', default=int(1e6), type=int)
    parser.add_argument('--num_frames', default=4, type=int, help='nframe, QNet input')
    parser.add_argument('--frame_size', default=84, type=int)
    parser.add_argument('--target_q_sync_interval', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--update_freq', default=4, type=int)
    parser.add_argument('--train_start_eps', default=1.0, type=float)
    parser.add_argument('--train_final_eps', default=0.1, type=float)
    parser.add_argument('--train_eps_num_steps', default=1000000, type=int)
    parser.add_argument('--eval_eps', default=0.01, type=float)
    parser.add_argument('--burn_in_steps', default=200000, type=int)
    parser.add_argument('--no_op_start', default=30, type=int)
    parser.add_argument('--use_double_dqn', action='store_true')
    parser.add_argument('--output', default='experiments/test')
    parser.add_argument('--algorithm', default='dqn', type=str)

    args = parser.parse_args()
    game_name = args.rom.split('/')[-1].split('.')[0]
    args.output = '%s_%s_%s' % (args.output, game_name, args.algorithm)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, 'configs.txt'), 'w') as f:
        print >>f, args

    random.seed(args.seed)
    np.random.seed(large_randint())
    torch.manual_seed(large_randint())
    torch.cuda.manual_seed(large_randint())
    return args


if __name__ == '__main__':
    args = main()
    torch.backends.cudnn.benckmark = True

    frame_skip = 4
    train_env = Environment(
        args.rom,
        frame_skip,
        args.num_frames,
        args.frame_size,
        args.no_op_start,
        large_randint(),
        True)
    eval_env = Environment(
        args.rom,
        frame_skip,
        args.num_frames,
        args.frame_size,
        args.no_op_start,
        large_randint(),
        False)

    replay_memory = ReplayMemory(args.replay_buffer_size)
    train_policy = LinearDecayGreedyEpsilonPolicy(
        args.train_start_eps, args.train_final_eps, args.train_eps_num_steps)
    eval_policy = GreedyEpsilonPolicy(args.eval_eps)
    optim_args = {
        'cls': torch.optim.Adam,
        'lr': 6.25e-5,
        'eps': 1.5e-4,
    }

    if 'dueling' == args.algorithm:
        QNClass = DuelingQNetwork
        AgentClass = DQNAgent
    elif 'pdueling' == args.algorithm:
        QNClass = PredDuelingQNetwork
        AgentClass = PredDQNAgent
    elif 'spdueling' == args.algorithm:
        QNClass = SinglePredDuelingQNetwork
        AgentClass = PredDQNAgent
    elif 'dqn' == args.algorithm:
        QNClass = DQNetwork
        AgentClass = DQNAgent
    else:
        assert False, '%s is not implemented yet' % args.algorithm

    q_net = QNClass(args.num_frames,
                    args.frame_size,
                    env.num_actions,
                    optim_args,
                    args.q_net)
    agent = AgentClass(q_net,
                       replay_memory,
                       args.gamma,
                       args.target_q_sync_interval,
                       args.use_double_dqn)
    eval_args = {
        'env': eval_env,
        'eval_per_iter': 100000,
        'policy': eval_policy,
        'num_eps': 20,
    }
    logger = Logger(os.path.join(args.output, 'train_log.txt'))

    agent.burn_in(env, args.burn_in_steps)

    train(
        train_env,
        train_policy,
        args.batch_size,
        args.num_iters,
        args.update_freq,
        eval_args,
        logger,
        args.output
    )

    # # fianl eval
    # eval_log = agent.eval(eval_env, eval_policy, 100)
    # print logger.log(eval_log)
