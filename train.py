"""Main DQN agent."""
import os
import time
import copy
import torch
import torch.nn
from torch.autograd import Variable
import numpy as np
import utils
from policy import GreedyEpsilonPolicy
import core
from logger import Logger


def train(agent,
          env,
          policy,
          replay_memory,
          gamma,
          batch_size,
          num_iters,
          steps_per_update,
          steps_per_sync,
          steps_per_eval,
          evaluator,
          output_dir):

    logger = Logger(os.path.join(output_dir, 'train_log.txt'))
    optim = torch.optim.Adam(agent.parameters(), lr=6.25e-5, eps=1.5e-4)
    action_dist = np.zeros(env.num_actions)

    best_avg_rewards = 0
    num_epsd = 0
    epsd_iters = 0
    epsd_rewards = 0
    t = time.time()
    for i in xrange(num_iters):
        if env.end:
            num_epsd += 1
            if num_epsd % 10 == 0:
                fps = epsd_iters / (time.time() - t)
                msg = 'Episode: %d, Iter: %d, Fps: %.2f'
                print logger.log(msg % (num_epsd, i+1, fps))
                print logger.log('sum clipped rewards %d' %  epsd_rewards)
                epsd_iters = 0
                epsd_rewards = 0
                t = time.time()

            state = env.reset()

        action = policy.get_action(state)
        action_dist[action] += 1
        next_state, reward = env.step(action)
        replay_memory.append(state, action, reward, next_state, env.end)
        state = next_state
        epsd_rewards += reward
        epsd_iters += 1

        if (i+1) %  steps_per_update == 0:
            # TODO, maybe: factor this out
            samples = replay_memory.sample(batch_size)
            states, actions, rewards, next_states, non_ends \
                = core.samples_to_tensors(samples)
            actions = utils.one_hot(actions.unsqueeze(1), agent.num_actions)
            targets = agent.compute_targets(rewards, next_states, non_ends, gamma)

            states = Variable(states)
            actions = Variable(actions)
            targets = Variable(targets)
            loss = agent.loss(states, actions, targets)
            loss.backward()
            optim.step()
            optim.zero_grad()
            logger.append('loss', loss.data[0])

            policy.decay()

        if (i+1) % steps_per_sync == 0:
            print 'syncing nets, i: %d' % (i+1)
            agent.sync_target()

        if (i+1) % steps_per_eval == 0:
            avg_rewards, eval_msg = evaluator()
            print logger.log(eval_msg)

            if avg_rewards > best_avg_rewards:
                prefix = os.path.join(output_dir, '')
                agent.save_q_net(prefix)


def evaluate(env, policy, num_epsd):
    state = env.reset()
    actions = np.zeros(env.num_actions)

    total_rewards = np.zeros(num_epsd)
    eps_idx = 0
    log = ''

    while eps_idx < num_epsd:
        action = policy.get_action(state)
        actions[action] += 1
        state, _ = env.step(action)

        if env.end:
            total_rewards[eps_idx] = env.total_reward
            eps_log = ('>>>Eval: [%d/%d], rewards: %s\n' %
                       (eps_idx+1, num_epsd, total_rewards[eps_idx]))
            log += eps_log
            if eps_idx < num_epsd - 1: # leave last reset to next run
                state = env.reset()
            eps_idx += 1

    avg_rewards =total_rewards.mean()
    eps_log = '>>>Eval: avg total rewards: %s\n' % avg_rewards
    log += eps_log
    log += '>>>Eval: actions dist:\n'
    probs = list(actions/actions.sum())
    for a, p in enumerate(probs):
        log += '\t action: %d, p: %.4f\n' % (a, p)

    return avg_rewards, log
