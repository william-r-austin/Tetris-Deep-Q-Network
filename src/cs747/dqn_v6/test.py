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

from cs747.dqn_v6.dqn_model import DeepQNetworkAtariSmall
from cs747.dqn_v6.experience_replay import ReplayMemory
from cs747.dqn_v6.resnet import Network as ResNet
from cs747.dqn_v6.tetris import Tetris
from cs747.dqn_v6.tetris_move_result import TetrisMoveResult
from cs747.dqn_v6 import format_util


def get_args():
    parser = argparse.ArgumentParser("""Test Tetris model""")
    parser.add_argument("--load_path", type=str, default="trained_models")
    parser.add_argument("--use-resnet", dest='use_resnet', action='store_true', help="Use ResNet when this option is present.")
    parser.set_defaults(use_resnet=False)
    parser.add_argument("--render-graphics", dest='render_graphics', action='store_true', help="Show graphics during testing.")
    parser.set_defaults(render_graphics=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    run_options = get_args()
    env = Tetris(render_flag=run_options.render_graphics)
    action_names = env.get_action_names()
    torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_dict = torch.load("{}".format(run_options.load_path))
    model = None
    if run_options.use_resnet:
        model = ResNet({
            'input_shape': (1, 1, env.height, env.width),
            'n_classes': len(action_names),
            'base_channels': 16,
            'block_type': 'basic',
            'depth': 110
        }).to(torch_device)
    else:
        model = DeepQNetworkAtariSmall(len(action_names)).to(torch_device)
    model.load_state_dict(model_dict['model_state_dict'])
    model.eval()

    game_number = 1

    while game_number <= 100:
        env.reset()
        
        while not env.gameover:
            board = env.get_current_board_state()
            current_state = format_util.to_dqn_84x84(board)
            
            with torch.no_grad():
                input_tensor = torch.unsqueeze(current_state, 0).to(torch_device)
                predictions = model(input_tensor)
                action_index = torch.argmax(predictions)
                env.do_action_by_id(action_index)
        
        print("Completed Game #{}, Score: {}, Tetrominoes: {}, Cleared Lines: {}, Action Count: {}"
              .format(game_number, env.score, env.tetrominoes, env.cleared_lines, env.action_count))
              
        game_number += 1
            
        
    
    