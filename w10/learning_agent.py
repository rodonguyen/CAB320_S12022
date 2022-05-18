# -*- coding: utf-8 -*-

"""
Modified on 2019-05-09

@author: f.maire@qut.edu.au

"""


import numpy as np
import random

import greyjack

class RL_Agent:

    # Constructor, initialize attributes
    def __init__(self, gamma = 0.9, n0=100):
        '''
        PARAM
          gamma : float : discount factor
          n0 : integer : parameter of the eps factor
        '''
        self.gamma = 0.9
        self.n0 = n0
        
        # initialize tables for (state, action) pairs occurrences, values
        self.N = np.zeros((10, 21, 2)) # dl, pl, actions (hit, stick)
        self.Q = np.zeros((10, 21, 2))
        
        # action index dictionary
        self.translator = {'hit':0, 'stick':1, 0:'hit', 1:'stick'}


    def eps_greedy_choice(self, state):
        '''
        Return the eps greedy action ('hit' or 'stick')
        Epsilon dependent on number of visits to the state        
        '''

        # collect visits
        visits_to_state = sum(self.N[state.dl_sum-1, state.pl_sum-1, :])

        # compute epsilon
        curr_epsilon = self.n0 / (self.n0 + visits_to_state)

        # epsilon greedy policy
        if random.random() < curr_epsilon:
            return 'hit' if random.random()<0.5 else 'stick'
        else:
            return self.translator[ np.argmax(self.Q[state.dl_sum-1, state.pl_sum-1, :]) ]


    def MC_learn(self, num_episodes = 10000, episode_window = 100, verbose=1):
        '''
        Play specified number of games, learning from experience using Monte-Carlo  
                   
        '''

        # Initialise
        game_outcomes = np.empty((num_episodes,))
        
        # Loop over episodes (complete game runs)
        for episode in range(num_episodes):

            # reset state action pair list
            episode_pairs = []

            # get initial state for current episode
            s = greyjack.State()
            s.do_first_round()

            # Execute until game ends
            is_game_over = False
            while not is_game_over:
                # choose action with epsilon greedy policy
                a = self.eps_greedy_choice(s) # action is a string
                # store action state pairs
                episode_pairs.append( (s, a) )
                # update visits
                self.N[s.dl_sum-1, s.pl_sum-1, self.translator[a]] += 1
                # execute action
                s, is_game_over = greyjack.step_greyjack(s, a)

            # Update Action value function accordingly
            r = s.compute_reward() # reward at the terminal state
            game_outcomes[episode] = r
            for s, a in episode_pairs:
                # learning rate decreases with number of  visits
                lr = 1.0  / (self.N[s.dl_sum-1, s.pl_sum-1, self.translator[a]])
#                lr = 0.001 # fixed learning rate
                error = r - self.Q[s.dl_sum-1, s.pl_sum-1, self.translator[a]]
	         self.Q[  # INSERT YOU CODE HERE (just one line)

            if verbose>0 and episode>=episode_window and episode%episode_window==0: 
                a,b  = episode-episode_window,episode
                print('Mean game payoff between episodes {} and {} is {}'.format(a,b,game_outcomes[a:b].mean()))
        
        return game_outcomes

    def SARSA_learn(self, num_episodes = 10000, episode_window = 100, verbose=1):
        '''
        Play specified number of games, learning from experience using Temporal Difference            
        '''
          # Initialise
        game_outcomes = np.empty((num_episodes,))
        
        # Loop over episodes (complete game runs)
        for episode in range(num_episodes):

	     # INSERT YOU CODE HERE (a few lines)



            if verbose>0 and episode>=episode_window and episode%episode_window==0: 
                a,b  = episode-episode_window,episode
                print('Mean game payoff between episodes {} and {} is {}'.format(a,b,game_outcomes[a:b].mean()))

        return game_outcomes
