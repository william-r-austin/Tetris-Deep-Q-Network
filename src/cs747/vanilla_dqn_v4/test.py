'''
Created on Apr 18, 2021

@author: William
'''

import argparse
import os
import shutil
from random import random, randint, sample, randrange
from datetime import datetime

import torch
import torch.nn as nn

from cs747.vanilla_dqn_v4.dqn_model import DeepQNetworkAtari
from cs747.vanilla_dqn_v4.experience_replay import ReplayMemory
from cs747.vanilla_dqn_v4.tetris import Tetris
from cs747.vanilla_dqn_v4.tetris_move_result import TetrisMoveResult
from cs747.vanilla_dqn_v4 import format_util


if __name__ == "__main__":
    model = torch.load("C:/Users/William/eclipse-workspace-cs747/Tetris-Deep-Q-Network/src/cs747/main/trained_models/tetris_30000")
    model.eval()
    
    env = Tetris()
    action_names = env.get_action_names()
    game_number = 1

    while game_number <= 100:
        env.reset()
        
        while not env.gameover:
            board = env.get_current_board_state()
            current_state = format_util.to_dqn_84x84(board)
            
            with torch.no_grad():
                input_tensor = torch.unsqueeze(current_state, 0)
                predictions = model(input_tensor)
                action_index = torch.argmax(predictions)
                env.do_action_by_id(action_index)
        
        print("Completed Game #{}, Score: {}, Tetrominoes: {}, Cleared Lines: {}, Action Count: {}"
              .format(game_number, env.score, env.tetrominoes, env.cleared_lines, env.action_count))
              
        game_number += 1
            
        
    
    