'''

Illustration of the use of a generator function to enumerate the solutions
of a Constraint Satisfaction Problem.


We assume that the set of constraints C of the problem are such 
if a partial assignment of the variables does not satisfy C, then no 
completion of the partial assignment can satisfy C. This will be the case
if C is a set of elementary expressions that do not include disjunctions.
The zebra puzzle falls in this category.

We will call such a set constraints, a "monotonic decreasing" set of constraints.

This property allows us to prune the search tree.



last modified 30/01/2022
by f.maire@qut.edu.au

'''

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
    True if my arbitrary constraints are satisfied
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
    md_constraint_search(my_toy_variables,
                                  my_toy_domains,
                                  my_toy_constraint_fn)