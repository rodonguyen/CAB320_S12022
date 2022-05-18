
'''
Solutions to Intro to ML Prac

    Last modified in 2022  by f.maire@qut.edu.au

'''


import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------

#  Solution to Exercises 1 and 2
#  This function works for x scalar and x ndarray

def polynom(x,w):
    '''
    Return  
       y = w[0]+w[1]  x**1 +  w[2] x**2 + ..+ w[n]  x**n
       where 
                n is the index of the last entry of w, 
               and y has the same shape as x  (scalar or ndarray)
    Params
      w : vector of the polynomial coefficients (n+1 elements)
      x  : numerical array 
    '''
    if isinstance(x, float) or isinstance(x, int):
        y = 0
    else:
        assert type(x) is np.ndarray
        y = np.zeros_like(x)
    for i in reversed(range(0,len(w))):
	# note the use of broadcasting
        y = w[i] + y*x
    return y
# ----------------------------------------

def test_polynom():
    print('---   test_polynom   ---')
    x = 2
    w = np.array([3,0,5])
    # compute 3+5*x**2 for x=2
    print (polynom(x,w))

# ----------------------------------------

# Exercise 3
# 
# the returned value can be use to compute 
# the value of a polynomial with   'w.dot(powerCol(x,n))'
#
def powerCol(x,n):
    '''
      Return 
         the column vector [1 x x^2 ... x^n]   if x is a scalar
             | 1    ..           1      |
             | x0   ..           xm     |
        y =  |       :                  |        if x is a vector x = [x0 x1 .. xm]
             |x**n  ..           xm**n  |
        
           y = w[0]+w[1]  x**1 +  w[2] x**2 + ..+ w[n]  x**n
           where import matplotlib.pyplot as plt

                    n is the index of the last entry of w, 
                   and y has the same shape as x  (scalar or ndarray)
    Params
      x  : numerical array
      n : integer
    '''
    # first we test whether x is a scalar
    # if this is the case, we transform is into a 1x1 matrix
    if isinstance(x, float) or isinstance(x, int):
        x = np.array([x])
    else:
        assert type(x) is np.ndarray
    # assert stops the program if a condition is not satisfied
    # we could have thrown an 'exception' instead
    p = np.ones((n+1,len(x)))
    for i in range(0,n):
        p[i+1] = p[i]*x
    return p


# ----------------------------------------

def test_powerCol():
    print('---   test_powerCol   ---')
    n = 5
    x = np.array([2,1,-3,4])
    print(powerCol(x,n))

# ----------------------------------------

# Exercise 4
def ex4():
    print('---   Ex4   ---')
    m = 50 # number of examples TRY DIFFERENT VALUES
    # x : training inputs
    x = np.linspace(-3,3,m)
    # t : training targets
    # 0.2 is arbitrary noise level
    t = np.sin(x) + 0.2 * np.random.randn(m) 


    n = 5 # degree of polynomial TRY DIFFERENT VALUES
    A = powerCol(x,n)
    # find best w
    w = np.linalg.pinv(A.T).dot( t.reshape((len(t),1)))
    print ('w = ', w)

    x_plot = np.linspace(-3,3,100)
    y_plot = polynom(x_plot,w)
    plt.plot(x, t, 'yo', label="data points")
    plt.plot(x_plot, np.sin(x_plot),'b-',label="target")
    plt.plot(x_plot, y_plot,'r-',label="approx")
    plt.legend(loc='upper left')
    plt.show()
    
# ----------------------------------------
    

    



# ----------------------------------------

if __name__ == "__main__":
    test_polynom()
    test_powerCol()
    ex4()


# ----------------------------------------

    
