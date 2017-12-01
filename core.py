"""Core classes."""
import utils
import random
import torch
import time
import numpy as np
from policy import GreedyEpsilonPolicy


class Sample(object):
    def __init__(self, state, action, reward, next_state, end):
        utils.assert_eq(type(state), type(next_state))

        self._state = (state * 255.0).astype(np.uint8)
        self._next_state = (next_state * 255.0).astype(np.uint8)
        self.action = action
        self.reward = reward
        self.end = end

    @property
    def state(self):
        return self._state.astype(np.float32) / 255.0

    @property
    def next_state(self):
        return self._next_state.astype(np.float32) / 255.0

    def __repr__(self):
        info = ('S(mean): %3.4f, A: %s, R: %s, NS(mean): %3.4f, End: %s'
                % (self.state.mean(), self.action, self.reward,
                   self.next_state.mean(), self.end))
        return info


class ReplayMemory(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.samples = []
        self.oldest_idx = 0

    def __len__(self):
        return len(self.samples)

    def _evict(self):
        """Simplest FIFO eviction scheme."""
        to_evict = self.oldest_idx
        self.oldest_idx = (self.oldest_idx + 1) % self.max_size
        return to_evict

    def burn_in(self, env, agent, num_steps):
        policy = GreedyEpsilonPolicy(1, agent) # uniform policy
        i = 0
        while i < num_steps or not env.end:
            if env.end:
                state = env.reset()
            action = policy.get_action(None)
            next_state, reward = env.step(action)
            self.append(state, action, reward, next_state, env.end)
            state = next_state
            i += 1
            if i % 10000 == 0:
                print '%d frames burned in' % i
        print '%d frames burned into the memory.' % i

    def append(self, state, action, reward, next_state, end):
        assert len(self.samples) <= self.max_size
        new_sample = Sample(state, action, reward, next_state, end)
        if len(self.samples) == self.max_size:
            avail_slot = self._evict()
            self.samples[avail_slot] = new_sample
        else:
            self.samples.append(new_sample)

    def sample(self, batch_size):
        """Simpliest uniform sampling (w/o replacement) to produce a batch.
        """
        assert batch_size < len(self.samples), 'no enough samples to sample from'
        return random.sample(self.samples, batch_size)

    def clear(self):
        self.samples = []
        self.oldest_idx = 0


def samples_to_tensors(samples):
    num_samples = len(samples)

    states_shape = (num_samples, ) + samples[0].state.shape
    states = np.zeros(states_shape, dtype=np.float32)
    next_states = np.zeros(states_shape, dtype=np.float32)

    rewards = np.zeros(num_samples, dtype=np.float32)
    actions = np.zeros(num_samples, dtype=np.int64)
    non_ends = np.zeros(num_samples, dtype=np.float32)

    for i, s in enumerate(samples):
        states[i] = s.state
        next_states[i] = s.next_state
        rewards[i] = s.reward
        actions[i] = s.action
        non_ends[i] = 0.0 if s.end else 1.0

    states = torch.from_numpy(states).cuda()
    actions = torch.from_numpy(actions).cuda()
    rewards = torch.from_numpy(rewards).cuda()
    next_states = torch.from_numpy(next_states).cuda()
    non_ends = torch.from_numpy(non_ends).cuda()

    return states, actions, rewards, next_states, non_ends
