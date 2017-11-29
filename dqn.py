"""Main DQN agent."""
import os
import time
import copy
import torch
import torch.nn as nn
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

    def parameters(self):
        return self.online_q_net.parameters()

    def target_q_values(self, states):
        q_vals = self.target_q_net(Variable(states, volatile=True)).data
        return q_vals

    def online_q_values(self, states):
        q_vals = self.online_q_net(Variable(states, volatile=True)).data
        return q_vals

    def sync_target(self):
        self.target_q_net = copy.deepcopy(self.online_q_net)

    def loss(self, states, actions, targets):
        """

        params:
            states: Variable [batch, channel, w, h]
            actions: Variable [batch, num_actions] one hot encoding
            targets: Variable [batch, 1]
        """
        utils.assert_eq(actions.size(1), self.num_actions)

        qs = self.online_q_net(states)
        preds = (qs * actions).sum(1)
        err = nn.functional.smooth_l1_loss(preds, targets)
        return err
