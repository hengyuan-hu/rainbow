"""Run Atari Environment with DQN."""
import os
import argparse
import torch
import model
import dqn
import utils
from env import Environment
from policy import GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from core import ReplayMemory
from train import train, evaluate


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari')
    parser.add_argument('--rom', default='roms/breakout.bin',
                        help='path to rom')
    parser.add_argument('--seed', default=10001, type=int,
                        help='Random seed')
    parser.add_argument('--q_net', default='', type=str,
                        help='load pretrained q net')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='discount factor')
    parser.add_argument('--num_iters', default=int(5e7), type=int)
    parser.add_argument('--replay_buffer_size', default=int(1e6), type=int)
    parser.add_argument('--frame_skip', default=4, type=int,
                        help='num frames for repeated action')
    parser.add_argument('--num_frames', default=4, type=int,
                        help='num stacked frames')
    parser.add_argument('--frame_size', default=84, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--frames_per_update', default=4, type=int)
    parser.add_argument('--frames_per_sync', default=32000, type=int)

    # for using eps-greedy exploration
    parser.add_argument('--train_start_eps', default=1.0, type=float)
    parser.add_argument('--train_final_eps', default=0.01, type=float)
    parser.add_argument('--train_eps_num_steps', default=int(1e6), type=int)

    # for noisy net
    parser.add_argument('--noisy_net', action='store_true')
    parser.add_argument('--sigma0', default=0.4, type=float)

    parser.add_argument('--eval_eps', default=0.001, type=float)
    parser.add_argument('--frames_per_eval', default=int(5e5), type=int)
    parser.add_argument('--burn_in_frames', default=200000, type=int)
    parser.add_argument('--no_op_start', default=30, type=int)

    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--output', default='exps/', type=str)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--double_dqn', action='store_true')
    parser.add_argument('--dueling', action='store_true')
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--num_atoms', default=51, type=int)

    parser.add_argument('--net', default=None, type=str)

    args = parser.parse_args()
    if args.dev:
        args.burn_in_frames = 500
        args.frames_per_eval = 5000
        args.output = 'devs/'

    game_name = args.rom.split('/')[-1].split('.')[0]

    model_name = []
    if args.noisy_net:
        model_name.append('noisy')

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
    utils.Config(vars(args)).dump(os.path.join(args.output, 'configs.txt'))
    return args


if __name__ == '__main__':
    args = main()

    torch.backends.cudnn.benckmark = True
    utils.set_all_seeds(args.seed)

    train_env = Environment(
        args.rom,
        args.frame_skip,
        args.num_frames,
        args.frame_size,
        args.no_op_start + 1,
        utils.large_randint(),
        True)
    eval_env = Environment(
        args.rom,
        args.frame_skip,
        args.num_frames,
        args.frame_size,
        args.no_op_start + 1,
        utils.large_randint(),
        False)

    if args.dist:
        assert not args.dueling, 'not supported yet.'
        q_net = model.build_distributional_basic_network(
            args.num_frames,
            args.frame_size,
            train_env.num_actions,
            args.num_atoms,
            args.noisy_net,
            args.sigma0,
            args.net)
        q_net.cuda()
        agent = dqn.DistributionalDQNAgent(
            q_net, args.double_dqn, train_env.num_actions, args.num_atoms, -10, 10)
    else:
        if args.dueling:
            q_net_builder = model.build_dueling_network
        else:
            q_net_builder = model.build_basic_network

        q_net = q_net_builder(
            args.num_frames,
            args.frame_size,
            train_env.num_actions,
            args.noisy_net,
            args.sigma0,
            args.net)

        q_net.cuda()
        agent = dqn.DQNAgent(q_net, args.double_dqn, train_env.num_actions)

    if args.noisy_net:
        train_policy = GreedyEpsilonPolicy(0, agent)
    else:
        train_policy = LinearDecayGreedyEpsilonPolicy(
            args.train_start_eps,
            args.train_final_eps,
            args.train_eps_num_steps,
            agent)

    eval_policy = GreedyEpsilonPolicy(args.eval_eps, agent)
    replay_memory = ReplayMemory(args.replay_buffer_size)
    replay_memory.burn_in(train_env, agent, args.burn_in_frames)

    evaluator = lambda logger: evaluate(eval_env, eval_policy, 10, logger)
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
          args.output)
