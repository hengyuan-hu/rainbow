import numpy as np
import utils
import torch


class GreedyEpsilonPolicy(object):
    def __init__(self, epsilon, q_agent):
        self.epsilon = float(epsilon)
        self.q_agent = q_agent

    def get_action(self, state):
        """Run Greedy-Epsilon for the given state.

        params:
            state: numpy-array [num_frames, w, h]

        return:
            action: int, in range [0, num_actions)
        """
        if np.random.uniform() <= self.epsilon:
            action = np.random.randint(0, self.q_agent.num_actions)
            return action

        state = torch.from_numpy(state)
        state = state.unsqueeze(0).cuda()

        q_vals = self.q_agent.online_q_values(state)
        utils.assert_eq(q_vals.size(0), 1)
        q_vals = q_vals.view(-1)
        q_vals = q_vals.cpu().numpy()
        action = q_vals.argmax()
        return action


class LinearDecayGreedyEpsilonPolicy(GreedyEpsilonPolicy):
    """Policy with a parameter that decays linearly.
    """
    def __init__(self, start_eps, end_eps, num_steps, q_agent):
        super(LinearDecayGreedyEpsilonPolicy, self).__init__(start_eps, q_agent)
        self.num_steps = num_steps
        self.decay_rate = (start_eps - end_eps) / float(num_steps)

    def decay(self):
        if self.num_steps > 0:
            self.epsilon -= self.decay_rate
            self.num_steps -= 1


# if __name__ == '__main__':
#     q_values = np.random.uniform(0, 1, (3,))
#     target_actions = q_values.argmax()

#     greedy_policy = GreedyEpsilonPolicy(0)
#     actions = greedy_policy(q_values)
#     assert (actions == target_actions).all()

#     uniform_policy = GreedyEpsilonPolicy(1)
#     uni_actions = uniform_policy(q_values)
#     assert not (uni_actions == target_actions).all()

#     steps = 9
#     ldg_policy = LinearDecayGreedyEpsilonPolicy(1, 0.1, steps)
#     expect_eps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1]
#     actual_eps = [1.0]
#     for i in range(steps+1):
#         actions = ldg_policy(q_values)
#         actual_eps.append(ldg_policy.epsilon)
#     assert (np.abs((np.array(actual_eps) - np.array(expect_eps))) < 1e-5).all()
