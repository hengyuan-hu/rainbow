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

    def save_q_net(self, prefix):
        torch.save(self.online_q_net.state_dict(), prefix+'online_q_net.pth')

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
            rewards: Tensor [batch]
            next_states: Tensor [batch, channel, w, h]
            non_ends: Tensor [batch]
            gamma: float
        """
        next_q_vals = self.target_q_values(next_states)

        if self.double_dqn:
            next_actions = self.online_q_values(next_states).max(1, True)[1]
            next_actions = utils.one_hot(next_actions, self.num_actions)
            next_qs = (next_q_vals * next_actions).sum(1)
        else:
            next_qs = next_q_vals.max(1)[0] # max returns a pair

        targets = rewards + gamma * next_qs * non_ends
        return targets

    def loss(self, states, actions, targets):
        """

        params:
            states: Variable [batch, channel, w, h]
            actions: Variable [batch, num_actions] one hot encoding
            targets: Variable [batch]
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

        z_vals = np.linspace(vmin, vmax, num_atoms).astype(np.float32)
        self.z_vals = Variable(torch.from_numpy(z_vals).unsqueeze(0)).cuda()

    def _q_values(self, q_net, states):
        """internal function to compute q_value

        params:
            q_net: self.online_q_net or self.target_q_net
            states: Variable [batch, channel, w, h]
        """
        probs = q_net(states) # [batch, num_actions, num_atoms]
        # print probs.size()
        # print self.z_vals.size()
        q_vals = (probs * self.z_vals).sum(2)
        return q_vals, probs

    def target_q_values(self, states):
        q_vals, _ = self._q_values(self.target_q_net, Variable(states, volatile=True))
        return q_vals.data

    def online_q_values(self, states):
        q_vals, _ = self._q_values(self.online_q_net, Variable(states, volatile=True))
        return q_vals.data

    def compute_targets(self, rewards, next_states, non_ends, gamma):
        """Compute batch of targets for distributional dqn

        params:
            rewards: Tensor [batch, 1]
            next_states: Tensor [batch, channel, w, h]
            non_ends: Tensor [batch, 1]
            gamma: float
        """
        assert not self.double_dqn, 'not supported yet'

        # get next distribution
        next_states = Variable(next_states, volatile=True)
        # [batch, num_actions], [batch, num_actions, num_atoms]
        next_q_vals, next_probs = self._q_values(self.target_q_net, next_states)
        next_actions = next_q_vals.data.max(1, True)[1] # [batch, 1]
        next_actions = utils.one_hot(next_actions, self.num_actions).unsqueeze(2)
        next_greedy_probs = (next_actions * next_probs.data).sum(1)

        # transform the distribution
        # print rewards.size()
        # print non_ends.size()
        # print self.z_vals.data.size()
        rewards = rewards.unsqueeze(1)
        non_ends = non_ends.unsqueeze(1)
        next_z_vals = rewards + gamma * non_ends * self.z_vals.data
        next_z_vals.clamp_(self.vmin, self.vmax)

        # project onto shared support
        b = (next_z_vals - self.vmin) / self.delta_z
        lower = b.floor()
        upper = b.ceil()
        # handle corner case where b is integer
        eq = (upper == lower).float()
        lower -= eq
        lt0 = (lower < 0).float()
        lower += lt0
        upper += lt0

        ml = (next_greedy_probs * (upper - b)).cpu().numpy()
        mu = (next_greedy_probs * (b - lower)).cpu().numpy()

        lower = lower.cpu().numpy().astype(np.int32)
        upper = upper.cpu().numpy().astype(np.int32)

        batch_size = rewards.size(0)
        mass = np.zeros((batch_size, self.num_atoms), dtype=np.float32)
        brange = range(batch_size)
        for i in range(self.num_atoms):
            mass[brange, lower[brange, i]] += ml[brange, i]
            mass[brange, upper[brange, i]] += mu[brange, i]

        return torch.from_numpy(mass).cuda()


    def loss(self, states, actions, targets):
        """

        params:
            states: Variable [batch, channel, w, h]
            actions: Variable [batch, num_actions] one hot encoding
            targets: Variable [batch, num_atoms]
        """
        utils.assert_eq(actions.size(1), self.num_actions)

        actions = actions.unsqueeze(2)
        probs = self.online_q_net(states) # [batch, num_actions, num_atoms]
        probs = (probs * actions).sum(1) # [batch, num_atoms]
        xent = -(targets * torch.log(probs)).sum(1)
        xent = xent.mean(0)
        return xent
