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


def train(agent,
          env,
          policy,
          replay_memory,
          gamma,
          batch_size,
          num_iters,
          steps_per_update,
          steps_per_sync,
          logger,
          steps_per_eval,
          evaluator,
          output_path):

    optim = torch.optim.Adam(agent.parameters(), lr=6.25e-5, eps=1.5e-4)

    num_epsd = 0
    epsd_iters = 0

    t = time.time()

    for i in xrange(num_iters):
        if env.end:
            num_epsd += 1
            if num_epsd % 100 == 0:
                fps = epsd_iters / (time.time() - t)
                msg = 'Episode: %d, Iter: %d, Fps: %.2f'
                print logger.log(msg % (num_epsd, i+1, fps))
                epsd_iters = 0
                t = time.time()
                # print policy.epsilon
            if num_epsd > 1:
                logger.append('rewards', rewards)

            state = env.reset()
            rewards = 0

        action = policy.get_action(state)
        # print action
        next_state, reward = env.step(action)
        replay_memory.append(state, action, reward, next_state, env.end)
        state = next_state
        rewards += reward
        epsd_iters += 1

        if (i+1) %  steps_per_update == 0:
            samples = replay_memory.sample(batch_size)
            states, actions, targets = core.samples_to_minibatch(
                samples, agent, gamma, True)
            states = Variable(states)
            actions = Variable(actions)
            targets = Variable(targets)
            loss = agent.loss(states, actions, targets)
            loss.backward()
            optim.step()
            optim.zero_grad()
            logger.append('loss', loss.data[0])

        if (i+1) % steps_per_sync == 0:
            print 'syncing nets, i: %d' % (i+1)
            agent.sync_target()

        if (i+1) % steps_per_eval == 0:
            eval_msg = evaluator()
            print logger.log(eval_msg)

        # if (i+1) % eval_args['eval_per_iter'] == 0:
        #     eval_log = evaluate(
        #         eval_args['env'], eval_args['policy'], eval_args['num_eps'])
        #     print logger.log(eval_log)

        # if (i+1) % (num_iters/4) == 0:
        #     model_path = os.path.join(
        #         output_path, 'net_%d.pth' % ((i+1)/(num_iters/4)))
        #     torch.save(agent.online_q_net.state_dict(), model_path)


def evaluate(env, policy, num_epsd):
    state_gpu = torch.cuda.FloatTensor(
        1, env.num_frames, env.frame_size, env.frame_size)

    state = env.reset()
    actions = np.zeros(env.num_actions)

    total_rewards = np.zeros(num_epsd)
    eps_idx = 0
    log = ''
    # TODO: implement no-op start
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

    eps_log = '>>>Eval: avg total rewards: %s\n' % total_rewards.mean()
    log += eps_log
    log += '>>>Eval: actions dist:\n'
    probs = list(actions/actions.sum())
    for a, p in enumerate(probs):
        log += '\t action: %d, p: %.4f\n' % (a, p)

    return log
