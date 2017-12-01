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
    def __init__(self, q_net, double_dqn, num_actions):
        self.online_q_net = q_net
        self.target_q_net = copy.deepcopy(q_net)
        self.double_dqn = double_dqn
        self.num_actions = num_actions

    def parameters(self):
        return self.online_q_net.parameters()

    def sync_target(self):
        self.target_q_net = copy.deepcopy(self.online_q_net)

    def target_q_values(self, states):
        q_vals = self.target_q_net(Variable(states, volatile=True)).data
        return q_vals

    def online_q_values(self, states):
        q_vals = self.online_q_net(Variable(states, volatile=True)).data
        return q_vals

    def compute_targets(self, rewards, next_states, non_ends, gamma):
        """Compute batch of targets for dqn

        params:
            rewards: Tensor [batch, 1]
            next_states: Tensor [batch, channel, w, h]
            non_ends: Tensor [batch, 1]
            gamma: float
        """
        target_q_vals = self.target_q_values(next_states)

        if self.double_dqn:
            next_actions = self.online_q_values(next_states).max(1, True)[1]
            next_actions = utils.one_hot(next_actions, self.num_actions)
            next_qs = (target_q_vals * next_actions).sum(1, True)
        else:
            next_qs = target_q_vals.max(1, keepdim=True)[0] # max returns a pair

        targets = rewards + gamma * next_qs * non_ends
        return targets

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


class DistributionalDQNAgent(DQNAgent):
    def __init__(self, q_net, double_dqn, num_actions, num_atoms, vmin, vmax):
        super(DistributionalDQNAgent, self).__init__(q_net, double_dqn, num_actions)

        self.num_atoms = num_atoms
        self.vmin = float(vmin)
        self.vmax = float(vmax)

        self.delta_z = (self.vmax - self.vmin) / (num_atoms - 1)

        z_vals = np.linspace(vmin, vmax, num_atoms)
        self.z_vals = torch.from_numpy(z_vals).view(1, 1, -1).cuda()

    def _q_values(self, q_net, states):
        """internal function to compute q_value

        params:
            q_net: self.online_q_net or self.target_q_net
            states: Variable [batch, channel, w, h]
        """
        probs = q_net(states) # [batch, num_actions, num_atoms]
        q_vals = (probs * z_vals).sum(2)
        return q_vals

    def target_q_values(self, states):
        q_vals = self._q_values(self.target_q_net, Variable(states, volatile=True))
        return q_vals.data

    def online_q_values(self, states):
        q_vals = self._q_values(self.online_q_net, Variable(states, volatile=True))
        return q_vals.data

    def compute_target(self, rewards, next_states, non_ends, gamma):
        pass

    def loss(self, states, actions, targets):
        pass
