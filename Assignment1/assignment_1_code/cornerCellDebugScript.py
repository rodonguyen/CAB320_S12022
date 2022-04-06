# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 08:36:09 2022

@author: gunzy
"""

import sokoban
import mySokobanSolver

def get_corner_pairs_old(warehouse, corners):
    corner_pairs = [] 
    for currentCorner in corners:
        for compareCorner in corners:

            # check if current corner is already paired up   CAN THIS BE DONE BETTER/EASIER???
            cornerPairFoundFlag = False
            for i in range(len(corner_pairs)):
                if currentCorner in corner_pairs[i]:
                    cornerPairFoundFlag = True
            if cornerPairFoundFlag:
                continue
            
            # no need to look at corners that are already the same
            if currentCorner == compareCorner:
                continue
            
            # corner pair found (along y axis), first have to check if a wall
            # is on the path before adding it to cornerPairs
            wall_on_path = False
            if currentCorner[0] == compareCorner[0]:
                # work out directin of range() function used next
                if currentCorner[1] > compareCorner[1]:
                    direction = -1;
                else:
                    direction = 1;
                for i in range(currentCorner[1],compareCorner[1],direction):
                    if (currentCorner[0], i) in warehouse.walls:
                        wall_on_path = True
                        break
                if not wall_on_path:
                    corner_pairs.append((currentCorner, compareCorner, 'y'))
                
            wall_on_path = False
            # corner pair found (along x axis)
            if currentCorner[1] == compareCorner[1]: # corner pair found
                if currentCorner[0] > compareCorner[0]:
                    direction = -1;
                else:
                    direction = 1;
                for i in range(currentCorner[0],compareCorner[0],direction):
                    if (i, currentCorner[1]) in warehouse.walls:
                        wall_on_path = True
                        break
                if not wall_on_path:
                    corner_pairs.append((currentCorner, compareCorner, 'x'))
                
    return corner_pairs

warehouse = sokoban.Warehouse()
w = "warehouses/warehouse_39.txt"
warehouse.load_warehouse(w)

inside_cells = mySokobanSolver.get_inside_cells(warehouse)
inside_corner_cells = mySokobanSolver.get_inside_corner_cells(warehouse, inside_cells)
corner_pairs = mySokobanSolver.get_corner_pairs(warehouse, inside_corner_cells)
corner_pairs_old = get_corner_pairs_old(warehouse, inside_corner_cells)


print()
print('Testing on:', w) 
print('Corners:', inside_corner_cells)
print('Corner pairs:', corner_pairs)
print('Corner pairs:', corner_pairs_old,  '(old version)')

