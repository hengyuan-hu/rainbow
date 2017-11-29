import matplotlib.pyplot as plt
import numpy as np
import sys


def render_avg_total_reward(file_name, output_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    avg_total_rewards = []
    for l in lines:
        if 'avg total rewards' in l:
            reward = float(l.split(' ')[-1])
            avg_total_rewards.append(reward)

    plt.plot(avg_total_rewards)
    plt.xlabel('Per 100K Iterations')
    plt.ylabel('Average Total Rewards')
    plt.savefig(output_name)
    plt.close()
    # plt.show()


def _get_avg_total_rewards(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    avg_total_rewards = []
    for l in lines:
        if 'avg total rewards' in l:
            reward = float(l.split(' ')[-1])
            avg_total_rewards.append(reward)
    return avg_total_rewards


def render_2avg_total_reward(file1, file2, output_name):
    rewards1 = _get_avg_total_rewards(file1)
    rewards2 = _get_avg_total_rewards(file2)

    print 'mean1:', np.mean(rewards1[-30:])
    print 'mean2:', np.mean(rewards2[-30:])
    print 'max1:', np.max(rewards1)
    print 'max2:', np.max(rewards2)

    plt.plot(rewards1, 'r-')
    plt.plot(rewards2, 'b-')
    plt.xlabel('Per 100K Iterations')
    plt.ylabel('Average Total Rewards')
    plt.savefig(output_name)
    plt.close()
    # plt.show()


def _get_clipped_rewards(file_name, length):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    rewards = []
    i = 0
    for l in lines:
        if ', Reward:' in l:
            reward = float(l.split(' ')[-3][:-1])
            if i % length == 0:
                rewards.append(reward)
            else:
                rewards[-1] += reward
            i += 1
    return rewards


def render_clipped_reward(file1, file2, output_name, length=100):
    rewards1 = _get_clipped_rewards(file1, length)
    rewards2 = _get_clipped_rewards(file2, length)

    print 'mean1:', np.mean(rewards1[-30:])
    print 'mean2:', np.mean(rewards2[-30:])

    plt.plot(rewards1, 'r-')
    plt.plot(rewards2, 'b-')
    plt.xlabel('Per %d Episodes' % length)
    plt.ylabel('X%d Clipped Rewards' % length)
    plt.savefig(output_name)
    plt.close()


def get_highest_rewards(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    max_reward = 0
    for l in lines:
        if '], rewards:' in l:
            reward = float(l.split(' ')[-1])
            max_reward = max(reward, max_reward)
    return max_reward


def compute_final_performance(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    final_rewards = []
    for l in lines:
        if '/100], rewards:' in l:
            reward = float(l.split(' ')[-1])
            final_rewards.append(reward)
    mean = np.mean(final_rewards)
    std = np.std(final_rewards)
    print 'final performance'
    print 'mean:', mean
    print 'std:', std


if __name__ == '__main__':

    log_files = sys.argv[1:3]
    output_file = sys.argv[3]

    render_2avg_total_reward(log_files[0], log_files[1], 'plt_'+output_file)
    render_clipped_reward(log_files[0], log_files[1], 'plt_train_'+output_file)

    print 'highest:', get_highest_rewards(log_files[0])
    print 'highest:', get_highest_rewards(log_files[1])

    # compute_final_performance(log_file)
