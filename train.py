import os
import time
import torch
import torch.nn
from torch.autograd import Variable
import numpy as np
import utils
from core import samples_to_tensors
from logger import Logger


def update_agent(agent, replay_memory, gamma, optim, batch_size):
    samples = replay_memory.sample(batch_size)
    states, actions, rewards, next_states, non_ends = samples_to_tensors(samples)
    actions = utils.one_hot(actions.unsqueeze(1), agent.num_actions)
    targets = agent.compute_targets(rewards, next_states, non_ends, gamma)
    states = Variable(states)
    actions = Variable(actions)
    targets = Variable(targets)
    loss = agent.loss(states, actions, targets)
    loss.backward()
    optim.step()
    optim.zero_grad()
    return loss.data[0]


def train(agent,
          env,
          policy,
          replay_memory,
          gamma,
          batch_size,
          num_iters,
          frames_per_update,
          frames_per_sync,
          frames_per_eval,
          evaluator,
          output_dir):

    logger = Logger(os.path.join(output_dir, 'train_log.txt'))
    optim = torch.optim.Adam(agent.parameters(), lr=6.25e-5, eps=1.5e-4)
    action_dist = np.zeros(env.num_actions)
    max_epsd_iters = 20000

    best_avg_rewards = 0
    num_epsd = 0
    epsd_iters = 0
    epsd_rewards = 0
    t = time.time()
    for i in xrange(num_iters):
        if env.end or epsd_iters > max_epsd_iters:
            num_epsd += 1
            if num_epsd % 10 == 0:
                fps = epsd_iters / (time.time() - t)
                logger.write('Episode: %d, Iter: %d, Fps: %.2f'
                             % (num_epsd, i+1, fps))
                logger.write('sum clipped rewards %d' %  epsd_rewards)
                logger.log()
                epsd_iters = 0
                epsd_rewards = 0
                t = time.time()

            state = env.reset()

        action = policy.get_action(state)
        action_dist[action] += 1
        next_state, reward = env.step(action)
        replay_memory.append(state, action, reward, next_state, env.end)
        state = next_state
        epsd_iters += 1
        epsd_rewards += reward

        if (i+1) % frames_per_update == 0:
            loss = update_agent(agent, replay_memory, gamma, optim, batch_size)
            logger.append('loss', loss)
            policy.decay()

        if (i+1) % frames_per_sync == 0:
            logger.write('>>>syncing nets, i: %d' % (i+1))
            agent.sync_target()

        if (i+1) % frames_per_eval == 0:
            logger.write('Train Action distribution:')
            for act, count in enumerate(action_dist):
                prob = float(count) / action_dist.sum()
                logger.write('\t action: %d, p: %.4f' % (act, prob))
            action_dist = np.zeros(env.num_actions)

            avg_rewards = evaluator(logger)
            if avg_rewards > best_avg_rewards:
                prefix = os.path.join(output_dir, '')
                agent.save_q_net(prefix)
                best_avg_rewards = avg_rewards


def evaluate(env, policy, num_epsd, logger):
    actions = np.zeros(env.num_actions)
    total_rewards = np.zeros(num_epsd)
    epsd_idx = 0
    epsd_iters = 0
    max_epsd_iters = 108000

    state = env.reset()
    while epsd_idx < num_epsd:
        action = policy.get_action(state)
        actions[action] += 1
        state, _ = env.step(action)
        epsd_iters += 1

        if env.end or epsd_iters >= max_epsd_iters:
            total_rewards[epsd_idx] = env.total_reward
            logger.write('>>>Eval: [%d/%d], rewards: %s' %
                         (epsd_idx+1, num_epsd, total_rewards[epsd_idx]))

            if epsd_idx < num_epsd - 1: # leave last reset to next run
                state = env.reset()

            epsd_idx += 1
            epsd_iters = 0

    avg_rewards = total_rewards.mean()
    logger.write('>>>Eval: avg total rewards: %s' % avg_rewards)
    logger.write('>>>Eval: actions dist:')
    probs = list(actions/actions.sum())
    for action, prob in enumerate(probs):
        logger.write('\t action: %d, p: %.4f' % (action, prob))

    return avg_rewards
