"""Run Atari Environment with DQN."""
import torch
import argparse
import os
import random
import numpy as np
from env import Environment
import dqn
from policy import GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
import model
from core import ReplayMemory
from train import train, evaluate



def large_randint():
    return random.randint(int(1e5), int(1e6))


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari')
    parser.add_argument('--rom', default='roms/breakout.bin', help='path to rom')
    parser.add_argument('--seed', default=10001, type=int, help='Random seed')
    parser.add_argument('--q_net', default='', type=str, help='load pretrained q net')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--num_iters', default=int(5e7), type=int)
    parser.add_argument('--replay_buffer_size', default=int(1e6), type=int)
    parser.add_argument('--num_frames', default=4, type=int, help='nframe, QNet input')
    parser.add_argument('--frame_size', default=84, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--frames_per_update', default=4, type=int)
    parser.add_argument('--frames_per_sync', default=32000, type=int)

    parser.add_argument('--train_start_eps', default=1.0, type=float)
    parser.add_argument('--train_final_eps', default=0.01, type=float)
    parser.add_argument('--train_eps_num_steps', default=int(1e6), type=int)
    parser.add_argument('--eval_eps', default=0.001, type=float)
    parser.add_argument('--frames_per_eval', default=int(5e5), type=int)

    parser.add_argument('--burn_in_frames', default=200000, type=int)
    parser.add_argument('--no_op_start', default=30, type=int)

    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--output', default='experiments/', type=str)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--double_dqn', action='store_true')
    parser.add_argument('--dueling', action='store_true')
    parser.add_argument('--dist', action='store_true')

    args = parser.parse_args()
    if args.dev:
        args.burn_in_frames = 500
        args.frames_per_eval = 5000

    game_name = args.rom.split('/')[-1].split('.')[0]

    model_name = []
    if args.dist:
        model_name.append('dist')

    if args.dueling:
        model_name.append('dueling')
    else:
        model_name.append('basic')

    if args.double_dqn:
        model_name.append('ddqn')

    if args.suffix:
        model_name.append(args.suffix)

    model_name = '_'.join(model_name)
    args.output = os.path.join(args.output, game_name, model_name)
    # TODO: better this
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

    if args.dist:
        num_atoms = 51
        q_net = model.build_distributional_basic_network(
            4, 84, train_env.num_actions, num_atoms, None).cuda()
        agent = dqn.DistributionalDQNAgent(
            q_net, args.double_dqn, train_env.num_actions, num_atoms, -10, 10)
    else:
        if args.dueling:
            q_net = model.build_dueling_network(4, 84, train_env.num_actions, None)
        else:
            q_net = model.build_basic_network(4, 84, train_env.num_actions, None)

        q_net.cuda()
        agent = dqn.DQNAgent(q_net, args.double_dqn, train_env.num_actions)

    train_policy = LinearDecayGreedyEpsilonPolicy(
        args.train_start_eps,
        args.train_final_eps,
        args.train_eps_num_steps,
        agent
    )
    eval_policy = GreedyEpsilonPolicy(args.eval_eps, agent)
    replay_memory = ReplayMemory(args.replay_buffer_size)
    replay_memory.burn_in(train_env, agent, args.burn_in_frames)

    evaluator = lambda : evaluate(eval_env, eval_policy, 10)
    train(agent,
          train_env,
          train_policy,
          replay_memory,
          args.gamma,
          args.batch_size,
          args.num_iters,
          args.frames_per_update,
          args.frames_per_sync,
          args.frames_per_eval,
          evaluator,
          args.output
    )
