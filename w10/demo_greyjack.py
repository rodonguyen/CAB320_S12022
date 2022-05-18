# -*- coding: utf-8 -*-
"""
Modified on 2017-09-11

@author: f.maire@qut.edu.au

"""


#import random
import numpy as np
import matplotlib.pyplot as plt

import learning_agent

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def test_monte_carlo(num_episodes=10**6, n0=1000):
    print ("Monte Carlo RL")
    print ("run for {} episodes".format(num_episodes))
    # learn
    agent = learning_agent.RL_Agent(gamma = 0.9, n0=n0)
    episode_window = int(num_episodes/100)
    win_hist = agent.MC_learn( num_episodes = num_episodes, episode_window = episode_window, verbose=1)
    
    win_hist_smooth = moving_average(win_hist,1000)

    plt.plot(range(len(win_hist_smooth)),win_hist_smooth,'-r')
    plt.ylabel('Average win over {} episodes'.format(episode_window))
    plt.xlabel('episode index')
    
    plt.title('Monte Carlo RL over {} episodes'.format(num_episodes))
    plt.show()


def test_sarsa(num_episodes=10**6, mlambda=None, n0=1000, avg_it=50):
    print ("SARSA RL")
    print ("run for {} episodes".format(num_episodes))
    # learn
    agent = learning_agent.RL_Agent(gamma = 0.9, n0=n0)
    episode_window = int(num_episodes/100)
    win_hist = agent.SARSA_learn(num_episodes = num_episodes, episode_window = episode_window, verbose=1)

    win_hist_smooth = moving_average(win_hist,1000)

    plt.plot(range(len(win_hist_smooth)),win_hist_smooth,'-r')
    plt.ylabel('Average win over {} episodes'.format(episode_window))
    plt.xlabel('episode index')
    
    plt.title('SARSA RL over {} episodes'.format(num_episodes))
    plt.show()



if __name__ == '__main__':
    test_monte_carlo()
#    test_sarsa()
