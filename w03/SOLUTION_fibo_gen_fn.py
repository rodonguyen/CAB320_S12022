#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:47:50 2022

Simple example of a generator function.

Illustrated with the Fibonacci sequence

@author: frederic
"""


# Recursive function to compute the nth element of the Fibonacci sequence
def fibo(n):
    '''
    Return the nth element of the Fibonacci sequence
    '''
    assert n>=0
    
    if n<=1:
        return n
    return fibo(n-2)+fibo(n-1)



def fibo_gen(s0, s1):
    '''
    Create a generator for a Fibonacci sequence

    Parameters
    ----------
    s0 : first value
    s1 : second value

    Returns
    -------
     a generator for a Fibonacci function starting with the values 
     s0 and s1

    '''
    yield s0
    yield s1
    prev, curr = s0, s1
    
    while True:
        s = prev+curr
        prev, curr = curr, s
        yield curr
        

if __name__ == '__main__':
    
    print(f'{fibo(35)=}\n')  # a few seconds!
    # print(f'{fibo(40)=}\n') # too long!
    
    for i, v in enumerate(fibo_gen(2,1)):
        print(f'S{i} = {v}')
        if i>12:
            # generate only the first fews elements of the sequence
            break