'''

Dynamic Programming prac sheet solution

The aim of this prac was to compare different implementations 
of the Levenshtein edit distance.

Created on Sun Oct  14 2018
Modified on Thu 18 2018
- fixed display of DP table
Modified  2019 Sept
- Added comments
- Fixed formatting bug in 'fill_lev_table' 
- Added reference to functools.lru_cache
Modified  2021 April
- changed completely to functools.lru_cache


@author: f.maire@qut.edu.au

'''

import timeit
import numpy as np
from functools import lru_cache



#    We can implement memoization with lru_cache from functools
#    
#    from functools import lru_cache
#    memoized_lev = lru_cache(maxsize=1024)(lev)
#    
#    We can also use the decorator notation
#    
#    @lru_cache(maxsize=1024)
#    def lev(a, 
#            b, 
#            insert_cost = lambda x:1 , 
#            delete_cost = lambda x:2 , 
#            match_cost = lambda x,y: 0 if x==y else 4):        
#    


def lev(a, 
        b, 
        insert_cost = lambda x:1 , 
        delete_cost = lambda x:2 , 
        match_cost = lambda x,y: 0 if x==y else 4):        
    '''
    Compute in a purely recursive fashion 
    the Levenshtein distance between 
    two sequences a and b
    @param
        a :  sequence
        b :  sequence
        insert_cost : insert cost function , 
        delete_cost : deletion cost function , 
        match_cost : match cost function                    
    '''

    if len(a)==0:
        # cost of inserting all elements of sequence b
        return sum([insert_cost(y) for y in b])
    if len(b)==0: 
        # cost of deleting all elements of sequence a
        return sum([delete_cost(x) for x in a])
    
    # the sequences a and b are non-empty
    return min(
        lev(a[:-1], b[:-1])+match_cost(a[-1],b[-1]) ,
        lev(a, b[:-1])+insert_cost(b[-1]) ,
        lev(a[:-1], b)+delete_cost(a[-1]) 
        )

# ----------------------------------------------------------------------------


# Memoized version of the 'lev' function
levm = lru_cache(maxsize=1024)(lev)


# ----------------------------------------------------------------------------

def fill_lev_table(a,b):
    '''
    Compute and display the Levenshtein edit distance table between  
    sequence 'a' and sequence 'b'
    @param
        a :  sequence
        b :  sequence
    '''
    
    # print head row (word b)
    print('\t'*2,end='')
    print(*(c for c in b),sep='\t')
    print('')    

    print('\t'*2,end='')
    print( *( levm('',b[:j+1]) for j in range(len(b)))  ,sep='\t')
    print('')    
    for i in range(len(a)):
        print(a[i],'\t',levm(a[:i+1],''),end='\t')
        for j in range(len(b)):
            print(levm(a[:i+1],b[:j+1]),end='\t')
        print('\n')    
    

# ----------------------------------------------------------------------------


# edit operation codes
dict_op = {0:'match', 1: 'insert', 2 :'delete',
           'match':0, 'insert':1, 'delete':2}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Exercise 3
def dynprog(x,
          y,
        insert_cost = lambda c:1 , 
        delete_cost = lambda c:2 , 
        match_cost = lambda cx, cy: 0 if cx==cy else 4):        
    '''
    Compute the Levenshtein distance between the  two sequences x and y 
    @param
        x :  sequence
        y :  sequence
        insert_cost : insert cost function , 
        delete_cost : deletion cost function , 
        match_cost = match cost function

    Compute the cost of editing sequence x into sequence y.
    Let nx , ny = len(x) , len(y)
    Sequence x is indexed from 0 to nx-1 (similar remark for y).
    M[nx,ny] is the cost of editing from x to y
    Note that M[0,3] is the cost of matching the empty string to the first 
    3 characters of sequence y.
    
    
    @return
    M,P
    where 
        M is the DP cost matrix
        M[i,j] : cost of matching x[:i] to y[:j]
                Note that x[i] and y[j] are not taken into account for M[i,j]
        M[nx,ny] : cost of matching x[:nx] to y[:ny]
        and
        P is the parent array to trace back the edit sequence
        P is used by the function 'explain_seq'
    '''
    
    #    x[0],...,x[nx-1]
    #    y[0],...,y[ny-1]
    #
    #    M : cost matrix

    nx = len(x)
    ny = len(y)
     
    # Cost matrix M
    # M[i,j] cost of matching the slice  x[:i] to the slice y[:j]
    # M[nx,ny] will be the cost of matching the whole sequences       
    M = np.zeros((nx+1,ny+1),dtype=float)

    # P[i,j] indicates to op code use for the last optimal operation
    # in matching the slice  x[:i] to the slice y[:j] 
    P = np.zeros((nx+1,ny+1),dtype=int) # parent 
    
    M[1:,0] = np.cumsum([delete_cost(c) for c in x])   
    P[1:,0] = dict_op['delete'] # delete op code 

    M[0,1:] = np.cumsum([insert_cost(c) for c in y])
    P[0,1:] = dict_op['insert'] # insert op code
    
    for ix in range(1,nx+1):
        for iy in range(1,ny+1):
            # print('ix {} iy {} '.format(ix,iy) )
            # M[ix][iy] cost of matching 
            # x[:ix] =x[0],..,x[ix-1 to y[:iy] = y[0],..,y[iy-1]
            L = [M[ix-1,iy-1] + match_cost(x[ix-1],y[iy-1]), # match x[ix-1] and  y[iy-1]
                 M[ix,iy-1] + insert_cost(y[iy-1]), # insert  y[iy-1]
                 M[ix-1,iy] + delete_cost(x[ix-1])] # delete  x[ix-1] 
            i_min = np.argmin(L)
            P[ix][iy] = i_min
            M[ix][iy] = L[i_min]
    return M,P


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    

# Exercise 3
def explain_dynprog(x,y,M,P):
    '''
    Retrieve the optimal sequence of edit operations given 
    the dyn prog tables M,P
    @pre
     M,P have been computed by 'dynprog'
    '''
    nx = len(x)
    ny = len(y)
    L =[]
    ix,iy = nx, ny
    while ix>0 and iy>0:
        if P[ix,iy]==0: #'match op':            
            L.append( ' match {} and {} '.format(x[ix-1],y[iy-1]))
            ix -= 1
            iy -= 1
        elif P[ix,iy]==1:# 'insert op'
            L.append( 'insert '+str(y[iy-1]))
            iy -= 1
        else: # 'delete op'
            L.append( 'delete '+str(x[ix-1]))
            ix -= 1
    #print('<A> ix = {} iy = {} '.format(ix,iy) )
    while ix>0:
        L.append( 'delete '+str(x[ix-1]))
        ix -= 1
    while iy>0:
        L.append( 'insert '+str(y[iy-1]))
        iy -= 1
    
    return list(reversed(L))
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

word_pair_list = [('abc','la'),
                  ('level','beaver'),
                  ('boating','whopping')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def exercise_1():
    '''
    Test the naive recursive implementation of the edit distance (function 'lev')
    '''        
    
    SETUP_STR = '''
from __main__ import lev
w1 = '{}'
w2 = '{}'
'''
    print('Exercise 1')
    num_executions = 50 # timeit param
    for w1,w2 in word_pair_list:
        running_time = timeit.timeit('lev(w1,w2)',
                            setup=SETUP_STR.format(w1,w2),
                            number=num_executions)
        print('Average running time for the pair ("{}","{}") is {} seconds\n'.format(w1,w2,running_time/num_executions))
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def exercise_2():
    '''
    Test the memoized implementation of the recursive edit distance (function 'lev')
    '''
    SETUP_STR = '''
from __main__ import levm
w1 = '{}'
w2 = '{}'
'''
    num_executions = 50 # timeit param
    print('Exercise 2')
    for w1,w2 in word_pair_list:
        running_time = timeit.timeit('levm(w1,w2)',  # levm is memoized lev
                            setup=SETUP_STR.format(w1,w2),
                            number=num_executions)
        print('Average running time for the pair ("{}","{}") is {} seconds\n'.format(w1,w2,running_time/num_executions))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def exercise_3():
    '''
    Compute the DP table and the sequence of edit actions
    '''
    
    # Some pairs of words
#    w1, w2= 'trump', 'dumbpark'
    w1, w2 = 'BOATING', 'AORTIC'
#    w1, w2 = 'pumpkin', 'plank'
#    w1, w2 = 'swimming', 'swings'
#    w1, w2 = 'EPISTEMIC', 'PLUMIST'
    
    
    print('w1 = {}'.format(w1))
    print('w2 = {}'.format(w2))

    M,P = dynprog(w1,w2, 
                  insert_cost = lambda c:1 , 
                  delete_cost = lambda c:2 , 
                  match_cost = lambda cx, cy: 0 if cx==cy else 4)    
        
    print('DP matrix')
    print(M)
    L = explain_dynprog(w1,w2,M,P)    
    print(L)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass

    exercise_1()
    exercise_2()
    exercise_3()




# ++++++++++++++++++++++ CODE CEMETARY ++++++++++++++++++++++ 

# def memoize(fn):

