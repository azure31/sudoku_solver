# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter


# Board consists of cells, rows, columns and squares(sub-grid)
# Each cell is represented by a 2D index indicating (row_idx, col_idx)



# Define hashmaps 
# id2square: gives corresponding sub-grid (1-9) based on divisibility of indexes of a cell by 3

id2square = {(0, 0):1, (0, 1):2, (0, 2):3, 
            (1, 0):4, (1, 1):5, (1, 2):6, 
            (2, 0):7, (2, 1):8, (2, 2):9}


# Define a dictionary that contains locations of all cells that belong to a sub-grid
# Sub-grids are from 1-9
square_dict = {k:[] for k in range(1, 10)}
for i in range(9):
    for j in range(9):
        square_dict[id2square[(i//3, j//3)]].append((i, j))


def get_square_peers(arr, i, j):
    
    ''' Given a board array and cell index, 
    find all the values in the sub-grid of the cell '''
    
    k = id2square[(i//3,j//3)]
    ids = square_dict[k].copy()
    ids.remove((i,j))
    return [arr[i] for i in ids]


def get_peers(arr, i, j):
    
    '''Get all neighbouring values of a cell with index (i,j) '''

    r_peers = list(arr[i, :j]) + list(arr[i, j+1:])
    c_peers = list(arr[:i, j]) + list(arr[i+1:, j])
    s_peers = get_square_peers(arr, i, j)
    return r_peers + c_peers + s_peers


def print_board(arr):
    
    ''' Print current board configuration '''
    
    for i in range(9):    
        if(i%3 == 0):
            print("-----------------------")
        row = "|".join(' ' + str(arr[i][j:j+3]) + ' ' for j in range(0, 9, 3))
        row = row.replace("[", "").replace("]", "").replace(",", " ")
        print(row)
    print("-----------------------")


def gen_cell_possibilities(arr, i, j):
    
    ''' Find all possible values for a cell based on values of its peers '''

    all_pos = set(range(1,10))
    peers = set(get_peers(arr, i, j))
    return list(all_pos - peers)


def gen_board_possibilities(arr):

    ''' Generate possible values for all empty cells in the board '''

    new_possible = {}
    for i in range(9):
        for j in range(9):
            if(arr[i, j] == 0):
                poss = gen_cell_possibilities(arr, i, j)
                new_possible[(i, j)] = poss

    return new_possible


def no_duplicates(items):

    ''' Check if any duplicate non-zero entries are present in items '''
    
    return all(items[k] == 1 for k in items if k>0)


def is_valid_board(arr):

    ''' Return True if current board configuration is not breaking any rules '''
    
    # Check for each row
    for i in range(9):
        items = Counter(arr[i, :])
        if(not no_duplicates(items)):
            return False
    
    # Check for each col
    for i in range(9):
        items = Counter(arr[:, i])
        if(not no_duplicates(items)):
            return False

    # Check for each square
    for k in square_dict:
        items = []
        keys = square_dict[k]
        for k in keys:
            items.append(arr[k])
        items = Counter(items)
        if(not no_duplicates(items)):
            return False
    return True


def update_board(arr, possible_dict):
    
    ''' Fill sure shot cases (only one-possible value for the cell) from 
    current board config, and test if board remains valid '''
        
    # Sure cases
    for k in possible_dict:
        if((len(possible_dict[k])==1)):
            arr[k] = possible_dict[k][0]
            if(not is_valid_board(arr)):
                arr[k] = 0
                return arr, False
            else:
                #print(f"Setting {k} as {arr[k]}")
                pass
            

    #print_board(arr)
    return arr, True


def pick_possibility(possible_dict):

    ''' Sort dict by number of possibilities, pick that cell where 
    least number of possibilities exist '''
    
    possible_dict = {k: v for k, v in sorted(possible_dict.items(), 
                    key=lambda item: len(item[1]))}
    
    key = list(possible_dict.keys())[0]
    key_vals = possible_dict[key]
    return key, key_vals


def solve_board(arr):
    
    ''' Recursively solve board using backtracking  '''
    
    if(not is_valid_board(arr)):
        return False

    # Generate possible values for empty cells
    curr_possible = gen_board_possibilities(arr)

    # Base Cases
    # If no empty cells
    if(len(curr_possible)==0):
        print("base case: sudoku solved")
        print_board(arr)
        return True
    
    # If an empty cells has no possible value
    if(any(len(curr_possible[k])==0 for k in curr_possible)):
        #print("Conflict")
        return False
    
    # Non-base case      
    # If any sure cell values based on current board state, then fill them
    if(any(len(curr_possible[k])==1 for k in curr_possible)):
       arr, is_valid = update_board(arr, curr_possible)
       
       if(is_valid):
           # Solve updated board state
           return solve_board(arr)
       else:
           return False
    
    # If no sure cell values, then pick a possible value and attempt to fill board        
    else:    
        key, key_vals = pick_possibility(curr_possible)    
        
        # Try out all possible values for cell with index key
        for k in key_vals:
            #print(f"Trying out {k} at {key}")
            arr_cpy = arr.copy()
            arr_cpy[key] = k
            if(solve_board(arr_cpy)):
                return True
        
        return False    
                
