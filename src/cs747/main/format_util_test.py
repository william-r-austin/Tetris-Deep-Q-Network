'''
Created on Apr 11, 2021

@author: William
'''
from cs747.sim.tetris import Tetris
from cs747.util import format_util
import torch

if __name__ == "__main__":
    env = Tetris()
    board = env.get_current_board_state()
    frame = format_util.to_dqn_84x84(board)
    board_tensor = torch.stack([frame, frame, frame, frame]).permute(1, 2, 0)
    
    print("Tetris Board:")
    print(board)
    
    print("\nBoard Tensor size:")
    print(board_tensor.shape)
    