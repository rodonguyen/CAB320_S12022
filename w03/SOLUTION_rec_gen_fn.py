'''

Illustration of the use of a generator function to enumerate the solutions
of a Constraint Satisfaction Problem.


    ** IMPORTANT CAVEAT **
      
    We assume that the set of constraints C of the problem are such 
    if a partial assignment of the variables does not satisfy C, then no 
    completion of the partial assignment can satisfy C. This will be the case
    if C is a set of elementary expressions that do not include disjunctions.
    The zebra puzzle falls in this category.
    
    We will call such a set constraints, a "monotonic decreasing" set of constraints.
    
    This property allows us to prune the search tree.



last modified 02/02/2022
by f.maire@qut.edu.au

'''

# << ----------------------- Toy CSP -----------------------------------

my_toy_variables = ('A','B','C')
my_toy_domains = ((False,True),  # domain for variable A 
           (-1,0,1),      # domain for variable B
           (1,2,3,4))     # domain for variable C



def my_toy_constraint_fn(**da):
    '''
    Test whether the a set of constraints is satisfied on 
    the assigned variables.  
    
    This set of constraints associated to this function is completely 
    arbitrary! You can write your own function.
    

    Parameters
    ----------
    da : dictionary
        Dictionary representing a partial assignment.
        For example, da = {'A':True, 'B':-1}

    Returns
    -------
    True if my totatlly arbitrary and meaningless constraints are satisfied
    False otherwise

    '''
    
    # print(f'{da=}') # debug

    if 'A' in da and da['A'] != True:
        return False    
    if 'B' in da and da['B'] != 1:
        return False
    if 'C' in da and da['C']+da['B']<4:
        return False
    
    return True
# >> ----------------------- Toy CSP -----------------------------------


# << ----------------------- Zebra Puzzle CSP -----------------------------------
 
# ('Color','Nationality','Drink','Smoke','Pet')
 

my_zebra_variables = ('red', 'green', 'ivory', 'yellow', 'blue' ,  #  Color 
           'Englishman', 'Spaniard', 'Ukranian', 'Japanese', 'Norwegian',  # Nationality
            'coffee', 'tea', 'milk', 'oj', 'water', # Drink
            'OldGold', 'Kools', 'Chesterfields', 'LuckyStrike', 'Parliaments', # Smoke
            'dog', 'snails', 'fox', 'horse', 'zebra')     # Pet

my_zebra_domains = tuple((1,2,3,4,5) for _ in range(len(my_zebra_variables)))

def my_zebra_constraint_fn(**da):
    '''
    Test whether the zebra puzzle constraints are satisfied by  
    a partial assignment.  
    
    Parameters
    ----------
    da : dictionary
        Dictionary representing a partial assignment of the zebra puzzle CSP.

    Returns
    -------
    True if the constraints are so far satisfied
    False otherwise

    '''
    
    # print(f'{da=}') # debug
    def imright(h1, h2):
        "House h1 is immediately right of h2 if h1-h2 == 1."
        return h1-h2 == 1

    def nextto(h1, h2):
        "Two houses are next to each other if they differ by 1."
        return abs(h1-h2) == 1
    
    def all_diff(*L):
        '''
            Check that the variables that are not None are different
        '''
        N = [v for v in L if not(v is None)]
        # N : list of the assigned variables
        return len(N)==len(set(N))
        
    [first,_,middle,_,_] = [1,2,3,4,5]
    
    # da.get() return None if the key does not belong to the dictionary da 
    
    # retrieve the values from the dictionary, assign them to local variables
    # of the CSP.  A variable is set to None if it is unassigned
    red, green, ivory, yellow, blue = [da.get(v) 
                    for  v in ('red', 'green', 'ivory', 'yellow', 'blue')]
    Englishman, Spaniard, Ukranian, Japanese, Norwegian = [da.get(v) 
                   for v in('Englishman', 'Spaniard', 'Ukranian', 'Japanese', 'Norwegian')]
    coffee, tea, milk, oj, water = [ da.get(v) 
                        for v in ('coffee', 'tea', 'milk', 'oj', 'water') ]
    OldGold, Kools, Chesterfields, LuckyStrike, Parliaments = [ da.get(v) 
                        for v in ('OldGold', 'Kools', 'Chesterfields', 'LuckyStrike', 'Parliaments')]
    dog, snails, fox, horse, zebra =  [ da.get(v) 
                        for v in ('dog', 'snails', 'fox', 'horse', 'zebra')]
     
    # Check that the variables have been assigned different values.
    # Consider first the color variables
    if not all_diff(red, green, ivory, yellow, blue):
        return False  
    # do the same test for the other groups of variables
    if not all_diff(Englishman, Spaniard, Ukranian, Japanese, Norwegian):
        return False
    if not all_diff (coffee, tea, milk, oj, water):
        return False
    if not all_diff (OldGold, Kools, Chesterfields, LuckyStrike, Parliaments)  :
        return False
    if not all_diff (dog, snails, fox, horse, zebra):
        return False
    
    # imright(green, ivory)
    # check a constraint only if all relevant variables have bee assigned    
    if green and ivory:         
        if not imright(green, ivory): # constraint 6
            return False
    
    # Englishman == red 
    if  Englishman and red:
        if  not (Englishman == red): # constraint 2
            return False

    # Norwegian == first           #10
    if Norwegian:
        if not (Norwegian == first):
            return False
        
    # nextto(Norwegian, blue)      #15
    if Norwegian and blue:
        if not nextto(Norwegian, blue):
            return False
        
    # coffee == green               #4
    if coffee and green:
        if not( coffee == green ):
            return False
        
    # Ukranian == tea              #5
    if Ukranian and tea:
        if not (Ukranian == tea):
            return False        

    # milk == middle               #9
    if milk:
        if not (milk == middle):
            return False        
        
    if Kools and yellow:              #8
        if not( Kools == yellow ):    #8
            return False
        
    if LuckyStrike and oj:            #13
        if not (LuckyStrike == oj):
            return False
        
    if Japanese and Parliaments:      #14
        if not (Japanese == Parliaments):      #14
            return False

    if Spaniard and dog:              #3
        if not( Spaniard == dog ):              #3
            return False

    if OldGold and snails:            #7
        if not(OldGold == snails):            #7
            return False
    
    if Chesterfields and fox:
        if not( nextto(Chesterfields, fox) ):
            return False
    
    if Kools and horse:
        if not (nextto(Kools, horse)):
            return False
  
    return True  # all the constraints satisfied


# >> ----------------------- Zebra Puzzle CSP -----------------------------------

def gen_satistactory_assignments(partial_assignment,
                    free_variables, 
                    free_domains, 
                    contraint_fn):
    '''
    Return a generator of satisfactory complete assignments that 
    are an extension of the assignment 'partial_assignment'
   
    PRE: 
        the partial assignment satisfies the constraints of the 
        "monotonic decreasing" boolean function 'contraint_fn'

    Parameters
    ----------
    partial_assignment : dictionary of the values of the variables that
                         have already been assigned
    free_variables : list of the unassigned variables
    free_domains : list of the domains of the unassigned variables
    contraint_fn : a monotonic decreasing" boolean function. 

    Yields
    ------
        a satistactory complete assignment in the form of a dictionary
        
    '''
    
    # defensive programming: consistency check
    assert len(free_variables)==len(free_domains)

    if len(free_variables)>0:
        # pick the first non-assigned variable
        var = free_variables[0]
        domain = free_domains[0]
        for val in domain:
            partial_assignment[var] = val
            if contraint_fn(**partial_assignment):
                for assignment in gen_satistactory_assignments(partial_assignment,
                                         free_variables[1:], 
                                         free_domains[1:], 
                                         contraint_fn):
                    yield assignment
                # restore partial assignment
                for v in free_variables:
                    if v in partial_assignment:
                        del partial_assignment[v]
    else:
        yield partial_assignment
                    
                
        
def md_constraint_search(variables, domains, contraint_fn):
    '''
    Enumerate the solutions of a Constraint Satisfaction Problem 
    where the set of constraints is monotonic decreasing.
    ''' 
    for x in gen_satistactory_assignments(
            dict(),
            variables, 
            domains, 
            contraint_fn):
        # print the solutions that have all variables assigned
        assert len(x) == len(variables)
        print(f'solution: {x=}')
        


if __name__ == '__main__':
    # md_constraint_search(my_toy_variables,
    #                               my_toy_domains,
    #                               my_toy_constraint_fn)
    
    md_constraint_search(my_zebra_variables,
                                  my_zebra_domains,
                                  my_zebra_constraint_fn)
    