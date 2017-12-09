import os
import time
import torch
import torch.nn
from torch.autograd import Variable
import numpy as np
import utils
import core
from logger import Logger


def update_agent(agent, replay_memory, gamma, optim, batch_size):
    samples = replay_memory.sample(batch_size)
    states, actions, rewards, next_states, non_ends = core.samples_to_tensors(samples)
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
        epsd_iters += 1
        epsd_rewards += reward

        if epsd_iters > frames_per_eval:
            print state.mean()
            print action

        if epsd_iters > frames_per_eval + 10:
            return

        if (i+1) % frames_per_update == 0:
            loss = update_agent(agent, replay_memory, gamma, optim, batch_size)
            logger.append('loss', loss)
            policy.decay()

        if (i+1) % frames_per_sync == 0:
            print 'syncing nets, i: %d' % (i+1)
            agent.sync_target()
            print 'epsd:', num_epsd, 'epsd_iter:', epsd_iters, 'rewards:', epsd_rewards

        if (i+1) % frames_per_eval == 0:
            print 'Train Action distribution:'
            for act, count in enumerate(action_dist):
                prob = float(count) / action_dist.sum()
                print '\t action: %d, p: %.4f' % (act, prob)
            action_dist = np.zeros(env.num_actions)

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

    eps_frames = 0
    max_eps_frames = 108000
    while eps_idx < num_epsd:
        action = policy.get_action(state)
        actions[action] += 1
        state, _ = env.step(action)
        eps_frames += 1

        if env.end or eps_frames >= max_eps_frames:
            total_rewards[eps_idx] = env.total_reward
            eps_log = ('>>>Eval: [%d/%d], rewards: %s\n' %
                       (eps_idx+1, num_epsd, total_rewards[eps_idx]))
            log += eps_log
            if eps_idx < num_epsd - 1: # leave last reset to next run
                state = env.reset()

            eps_idx += 1
            eps_frames = 0

    avg_rewards = total_rewards.mean()
    eps_log = '>>>Eval: avg total rewards: %s\n' % avg_rewards
    log += eps_log
    log += '>>>Eval: actions dist:\n'
    probs = list(actions/actions.sum())
    for action, prob in enumerate(probs):
        log += '\t action: %d, p: %.4f\n' % (action, prob)

    return avg_rewards, log
