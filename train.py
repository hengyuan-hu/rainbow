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


def train(agent, env, policy, batch_size, num_iters, update_freq,
          eval_args, logger, output_path):
    num_eps = 0
    prev_eps_iters = 0
    state_gpu = torch.cuda.FloatTensor(
        1, env.num_frames, env.frame_size, env.frame_size)

    for i in xrange(num_iters):
        if env.end:
            t = time.time()
            state = env.reset()
            rewards = 0

        state_gpu.copy_(torch.from_numpy(state.reshape(state_gpu.size())))
        action = agent.select_action(state_gpu, policy)
        next_state, reward = env.step(action)
        agent.replay_memory.append(state, action, reward, next_state, env.end)
        state = next_state
        rewards += reward

        if (i+1) % update_freq == 0:
            agent._update_q_net(batch_size, logger)

        if (i+1) % (update_freq * agent.target_update_freq) == 0:
            agent.target_q_net = copy.deepcopy(agent.online_q_net)

        if env.end:
            fps = (i+1-prev_eps_iters) / (time.time()-t)
            num_eps += 1
            log_msg = ('Episode: %d, Iter: %d, Reward: %d; Fps: %.2f'
                       % (num_eps, i+1, rewards, fps))
            print logger.log(log_msg)
            prev_eps_iters = i+1

        if (i+1) % eval_args['eval_per_iter'] == 0:
            eval_log = evaluate(
                eval_args['env'], eval_args['policy'], eval_args['num_eps'])
            print logger.log(eval_log)

        if (i+1) % (num_iters/4) == 0:
            model_path = os.path.join(
                output_path, 'net_%d.pth' % ((i+1)/(num_iters/4)))
            torch.save(agent.online_q_net.state_dict(), model_path)


def evaluate(agent, env, policy, num_eps):
    """Test your agent with a provided environment.

    You can also call the render function here if you want to
    visually inspect your policy.
    """
    state_gpu = torch.cuda.FloatTensor(
        1, env.num_frames, env.frame_size, env.frame_size)
    state = env.reset()
    actions = np.zeros(env.num_actions)

    total_rewards = np.zeros(num_eps)
    eps_idx = 0
    log = ''
    while eps_idx < num_eps:
        state_gpu.copy_(torch.from_numpy(state.reshape(state_gpu.size())))
        action = agent.select_action(state_gpu, policy)
        actions[action] += 1
        state, _ = env.step(action)

        if env.end:
            total_rewards[eps_idx] = env.total_reward
            eps_log = ('>>>Eval: [%d/%d], rewards: %s\n' %
                       (eps_idx+1, num_eps, total_rewards[eps_idx]))
            log += eps_log
            if eps_idx < num_eps-1: # leave last reset to next run
                state = env.reset()
            eps_idx += 1

    eps_log = '>>>Eval: avg total rewards: %s\n' % total_rewards.mean()
    log += eps_log
    log += '>>>Eval: actions dist: %s\n' % list(actions/actions.sum())
    return log
