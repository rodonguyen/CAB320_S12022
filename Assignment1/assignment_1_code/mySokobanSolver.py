'''

    Sokoban assignment


The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

You are NOT allowed to change the defined interfaces.
In other words, you must fully adhere to the specifications of the 
functions, their arguments and returned values.
Changing the interfacce of a function will likely result in a fail 
for the test of your code. This is not negotiable! 

You have to make sure that your code works with the files provided 
(search.py and sokoban.py) as your code will be tested 
with the original copies of these files. 

Last modified by 2022-03-27  by f.maire@qut.edu.au
- clarifiy some comments, rename some functions
  (and hopefully didn't introduce any bug!)

'''

# You have to make sure that your code works with 
# the files provided (search.py and sokoban.py) as your code will be tested 
# with these files
import search 
import sokoban

X_INDEX = 0
Y_INDEX = 1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(10210776, 'Mitchell', 'Egan'), (10396489, 'Jaydon', 'Gunzburg'), (10603280, 'Dac Duy Anh', 'Nguyen')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_inside_cells(warehouse, inside_cells = [], cell = None):
    '''
    Recursively identify inside cells (the cells that are enclosed by walls in 
    the space of the worker) using the flood fill algorithm. Adapted from:
    https://en.wikipedia.org/wiki/Flood_fill#Stack-based_recursive_implementation_(four-way)

    Parameters
    ----------
    warehouse : Warehouse
        A Warehouse object with the worker inside the warehouse.
    inside_cells : List, optional
        The list of already identified inside cells (coordinate tuples). The default is [].
    cell : Tuple, optional
        The cell to check. The default is None.

    Returns
    -------
    inside_cells : List
        The list of identified inside cells.
    '''
    
    if (cell is None):
        cell = warehouse.worker
    
    if (cell in inside_cells or cell in warehouse.walls):
        return inside_cells
    
    inside_cells.append(cell)
    
    # Recursively call get_inside_cells on cells to the north, south, west and east of current position
    inside_cells = get_inside_cells(warehouse, inside_cells, (cell[X_INDEX], cell[Y_INDEX] + 1))
    inside_cells = get_inside_cells(warehouse, inside_cells, (cell[X_INDEX], cell[Y_INDEX] - 1))
    inside_cells = get_inside_cells(warehouse, inside_cells, (cell[X_INDEX] - 1, cell[Y_INDEX]))
    inside_cells = get_inside_cells(warehouse, inside_cells, (cell[X_INDEX] + 1, cell[Y_INDEX]))

    return inside_cells

def is_corner(warehouse, cell):
    '''
    Parameters
    ----------
    warehouse : TYPE
        DESCRIPTION.
    cell : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    wall_neighbour_x = (cell[X_INDEX] - 1, cell[Y_INDEX]) in warehouse.walls or (cell[X_INDEX] + 1, cell[Y_INDEX]) in warehouse.walls
    wall_neighbour_y = (cell[X_INDEX], cell[Y_INDEX] - 1) in warehouse.walls or (cell[X_INDEX], cell[Y_INDEX] + 1) in warehouse.walls
    
    return wall_neighbour_x and wall_neighbour_y

def get_inside_corner_cells(warehouse, inside_cells):
    '''
    First finds all corner cells, then removes one that are not inside the warehouse
    (known from get_inside_cells())

    Parameters
    ----------
    warehouse : Warehouse
        A Warehouse object with the worker inside the warehouse.

    Returns
    -------
    corner_cells : List
        The list of identified corner cells.
    '''
    inside_corner_cells = []
    
    for inside_cell in inside_cells:
        if is_corner(warehouse, inside_cell):
            inside_corner_cells.append(inside_cell)
                
    return inside_corner_cells

def get_corner_pairs(warehouse, corners):
    '''
    returns a list of pairs of corners that are opposite each other,
    and the axis direction the pair is on ('x' or 'y')

    Parameters
    ----------
    warehouse : Warehouse
        A Warehouse object with the worker inside the warehouse.

    Returns
    -------
    corners : 3-tuple List
        List of identified corner pairs
    '''

    corner_pairs = []

    for one in range(len(corners)-1):
        for two in range(one+1, len(corners)):
            corner_1 = corners[one]
            corner_2 = corners[two]

            # Finding corner pair along Y axis
            wall_on_path = False
            # If they are on the same X
            if corner_1[X_INDEX] == corner_2[X_INDEX]:
                direction = -1 if corner_1[Y_INDEX] > corner_2[Y_INDEX] else 1

                # Check if there is wall on the path from corner_1 to corner_2
                for i in range(corner_1[Y_INDEX], corner_2[Y_INDEX]+1, direction):
                    if (corner_1[X_INDEX], i) in warehouse.walls:
                        wall_on_path = True
                        break
                if not wall_on_path:
                    corner_pairs.append((corner_1, corner_2, 'y'))

            # Finding corner pair along X axis
            wall_on_path = False
            # If they are on the same Y
            if corner_1[Y_INDEX] == corner_2[Y_INDEX]:
                direction = -1 if corner_1[X_INDEX] > corner_2[X_INDEX] else 1

                # Check if there is wall on the path from corner_1 to corner_2
                for i in range(corner_1[X_INDEX], corner_2[X_INDEX]+1, direction):
                    if (i, corner_1[Y_INDEX]) in warehouse.walls:
                        wall_on_path = True
                        break
                if not wall_on_path:
                    corner_pairs.append((corner_1, corner_2, 'x'))
    return corner_pairs


def taboo_cells(warehouse):
    '''  
    Identify the taboo cells of a warehouse. A "taboo cell" is by definition
    a cell inside a warehouse such that whenever a box get pushed on such 
    a cell then the puzzle becomes unsolvable. 
    
    Cells outside the warehouse are not taboo. It is a fail to tag an 
    outside cell as taboo.
    
    When determining the taboo cells, you must ignore all the existing boxes, 
    only consider the walls and the target  cells.  
    Use only the following rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of 
             these cells is a target.
    
    @param warehouse: 
        a Warehouse object with the worker inside the warehouse

    @return
       A string representing the warehouse with only the wall cells marked with 
       a '#' and the taboo cells marked with a 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''
    
    inside_cells = get_inside_cells(warehouse)
    inside_corner_cells = get_inside_corner_cells(warehouse, inside_cells)
    
    return inside_corner_cells

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. 
    
    '''
    
    #
    #         "INSERT YOUR CODE HERE"
    #
    #     Revisit the sliding puzzle and the pancake puzzle for inspiration!
    #
    #     Note that you will need to add several functions to 
    #     complete this class. For example, a 'result' method is needed
    #     to satisfy the interface of 'search.Problem'.
    #
    #     You are allowed (and encouraged) to use auxiliary functions and classes

    
    def __init__(self, warehouse):
        raise NotImplementedError()

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        """
        raise NotImplementedError

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action_seq(warehouse, action_seq):
    '''
    
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Impossible', if one of the action was not valid.
           For example, if the agent tries to push two boxes at the same time,
                        or push a box into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    
    ##         "INSERT YOUR CODE HERE"
    
    raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban(warehouse):
    '''
    This function analyses the given warehouse.
    It returns the two items. The first item is an action sequence solution. 
    The second item is the total cost of this action sequence.
    
    @param 
     warehouse: a valid Warehouse object

    @return
    
        If puzzle cannot be solved 
            return 'Impossible', None
        
        If a solution was found, 
            return S, C 
            where S is a list of actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
            C is the total cost of the action sequence C

    '''
    
    raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

