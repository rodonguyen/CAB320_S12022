'''
A function to simulate a roulette.
Useful for proportianl selection in GA

author f.maire@qut.edu.au

'''
import numpy as np



def fitness_prop_select(F):
    '''
    Selet an individual randomly proportionally to its fitness

    Parameters
    ----------
    F : 1D array of fitness values of the population
        DESCRIPTION.

    Returns
    -------
    i  : index of the selected individual

    '''
    C = np.cumsum(F)
    r = np.random.rand()*C[-1] # pick a random value between 0 and the sum of the fitness values
    A = C>r # boolean
    i = A.nonzero()[0][0] # extract the index of the first true entry
    return i


# some test

F = np.array([4,3,20,2,1,5])

i = fitness_prop_select(F)
print(i)
