'''
Created on Apr 18, 2021

@author: William
'''
import torch
import numpy as np

def to_dqn_simple(tetris_board):
    board_np = np.asarray(tetris_board, np.float32, 'C')
    board_np_clipped = np.clip(board_np, -1.0, 1.0)
    board_tensor = torch.flatten(torch.from_numpy(board_np_clipped))
    return board_tensor
    

def to_dqn_84x84(tetris_board):
    target_size = 84
    frame_tensor = torch.zeros(target_size, target_size, dtype=torch.float)
    height = len(tetris_board)
    width = len(tetris_board[0])
    
    for target_row in range(target_size):
        for target_col in range(target_size):
            src_row = int(target_row * height / target_size)
            src_col = int(target_col * width / target_size)
            src_value = tetris_board[src_row][src_col]
            if src_value != 0:
                src_value = 1 if src_value > 0 else -1
            
            frame_tensor[target_row, target_col] = src_value
    #print("Frame tensor shape")
    #print(frame_tensor.size())
    
    board_tensor = torch.stack([frame_tensor, frame_tensor, frame_tensor, frame_tensor])
    #.permute(1, 2, 0)
    return board_tensor
            
    
    # TODO - Make this more efficient by computing a
    #        slice for each non-zero point.
    '''    
    for row in range(len(tetris_board)):
        for col in range(len(tetris_board[0])):
            if tetris_board[row][col] != 0:
                #board_tensor[a:b][c:d]
    ''' 
    
    