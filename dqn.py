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


class DQNAgent(object):
    def __init__(self, q_net, num_actions):
        self.online_q_net = q_net
        self.target_q_net = copy.deepcopy(q_net)
        self.num_actions = num_actions

    def burn_in(self, env, num_burn_in):
        # TODO: move out to Replay Buffer
        policy = GreedyEpsilonPolicy(1) # uniform policy
        dummy_q_values = np.zeros(env.num_actions)
        i = 0
        while i < num_burn_in or not env.end:
            if env.end:
                state = env.reset()
            action = policy(dummy_q_values)
            next_state, reward = env.step(action)
            self.replay_memory.append(state, action, reward, next_state, env.end)
            state = next_state
            i += 1
            if i % 10000 == 0:
                print '%d frames burned in' % i
        print '%d frames burned into the memory.' % i

    def target_q_values(self, states):
        q_vals = self.target_q_net(Variable(states, volatile=True)).data
        return q_vals

    def online_q_values(self, states):
        q_vals = self.online_q_net(Variable(states, volatile=True)).data
        return q_vals

    def loss(self, states, actions, target):
        """

        params:
            states: Variable [batch, channel, w, h]
            actions: Variable [batch, num_actions] one hot encoding
            targets: Variable [batch, 1]
        """
        # utils.assert_eq(a.dim(), 2)
        utils.assert_eq(actions.size(1), self.num_actions)

        qs = self.online_q_net(states)
        preds = (qs * actions).sum(1)
        err = nn.functional.smooth_l1_loss(pred, tagets)
        return err

    # def _update_q_net(self, batch_size, logger):
    #     samples = self.replay_memory.sample(batch_size)
    #     x, a, y = core.samples_to_minibatch(samples, self)
    #     loss = self.online_q_net.train_step(x, a, y, 10)
    #     logger.append('loss', loss)
