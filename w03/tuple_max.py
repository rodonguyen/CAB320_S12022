#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Manipulation of nested tuples of integers

Consider a nested tuple T. For example,

    T =  ((10, -8), ((-9, 3, 6, -7), (6,), (2, 7), 6))

You can get an indented display of its associated tree 
with a call to print_nested_tuple(T) that will output in this case

    || 10
    || -8
    |
    ||| -9
    ||| 3
    ||| 6
    ||| -7
    ||
    ||| 6
    ||
    ||| 2
    ||| 7
    ||
    || 6
    |

The function 'generate_nested_tuple' creates random nested tuples

The function 'get_max' compute the maximum of a nested tuple and its location

Last modified on 2022/01/31  by f.maire@qut.edu.au

"""

from cmath import inf
from random import randint


def generate_nested_tuple(max_depth=4, max_branch=4, max_abs_value=10):
    '''
    Create and return a nested tuple of integers.    
    
    @param
        max_depth: upper bound for the depth of the associated tree
        max_branch: upper bound for the brancing factor of the associated tree
        max_abs_value: upper bound for the abs values of the associated tree
    '''
    if max_depth==0:
        return randint(-max_abs_value,max_abs_value)
    else:
        return tuple( 
                generate_nested_tuple(randint(0,max_depth-1), max_branch, max_abs_value)
                for _ in range(randint(1,max_branch))
                    )

##########################################################################################

temp = []
I = []
v = -inf
temp_location = []

def get_max(T):
    global v, temp_location, temp, I
    
    print('T =', T)

    for location, t in enumerate(T):
        if isinstance(t, tuple):
            temp.append(location)
            get_max(t)
        else: 
            if v < t:
                v = t
                temp.append(location)
                I = temp.copy()
                print('------> new I =', I)
                temp.pop(-1)

    if len(temp) != 0: 
        temp.pop(-1)
    
    return v, I
            



def print_nested_tuple(T, margin=''):
    if not (type(T) == tuple):
        # T is a scalar
        print(margin,T)
    else: # scan the top level
        for t in T:
            print_nested_tuple(t, margin=margin+'|')
        print(margin)
    
def test_1():
    '''
    Test the function 'get_max' on an example
    '''
    T  = ((-2, (-6, -1, 7, 1)), (8, -8), ((2, -5), -6, (-3, -5, -6, -3), (-4,)))
    v_max, I_max = get_max(T)
    print('T = ' ,T)
    print('v_max = ', v_max)
    print('I_max = ', I_max)
    assert v_max== 8
    assert I_max == [1,0]
   
def test_2():
    '''
    Test the function 'get_max' on a random example
    '''

    T  = generate_nested_tuple(max_depth=5)
    v_max, I_max = get_max(T)
    print('T = ' ,T)
    print('v_max = ', v_max)
    print('I_max = ', I_max)
    
    
def test_3():
    '''
    Test the function 'print_nested_tuple' on a random example
    '''
    T  = generate_nested_tuple(max_depth=5)
#    T =  (((3, -6, 7), 6, -2, (-2,)), -10)
#    T =  (((3,), -6, 7), 6, (-2,4,), -10)
#    T =  (((3,), -6, 7), 6, (-2,),(4,), -10)
    T =  (((5,),), ((5, -3, 9, -1),), (-1, ((2,), (-9, 7, -10)), (4, 1), (-8, 1, -2, -7)))    
        
    print('T = ' ,T)
    print('-------')
    print_nested_tuple(T)
    
def test_4():
    T = ((3,8,12), 5, (7,9, (10,14)))
    # T = (1, (10,14))

    v_max, I_max = get_max(T)
    print('T = ' ,T)
    print('v_max = ', v_max)
    print('I_max = ', I_max)
    print('-------')
    print_nested_tuple(T)
    
if __name__ == '__main__':
    # test_1()
    # test_2()
    # test_3()
    test_4()
