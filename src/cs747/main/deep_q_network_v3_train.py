"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample, randrange

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from cs747.models.deep_q_network import DeepQNetworkAtari
from cs747.util.experience_replay import ReplayMemory
from cs747.sim.tetris import Tetris
from collections import deque
from cs747.sim.tetris_move_result import TetrisMoveResult
from cs747.util import format_util

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--num_decay_episodes", type=int, default=2000)
    parser.add_argument("--num_episodes", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=1000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args

def choose_explore_or_exploit():
    return "EXPLORE"

def get_tensor_for_state(board_state):
    return format_util.to_dqn_84x84(board_state)

def do_minibatch_update(episode, epoch, game_id, env, model, replay_memory, optimizer, criterion, batch_size, gamma):
    batch = replay_memory.get_random_sample(batch_size)
    state_batch = torch.stack(tuple(sample.begin_tensor for sample in batch))
    next_state_batch = torch.stack(tuple(sample.next_tensor for sample in batch))
    
    if torch.cuda.is_available():
        state_batch = state_batch.cuda()
        next_state_batch = next_state_batch.cuda()
    
    reward_list = [sample.reward for sample in batch]
    game_active_list = [(0 if sample.final_state_flag else 1) for sample in batch]
    reward_tensor = torch.tensor(reward_list)
    game_active_tensor = torch.tensor(game_active_list)

    if torch.cuda.is_available():
        reward_tensor = reward_tensor.cuda()
        game_active_tensor = game_active_tensor.cuda()

    model.eval()
    with torch.no_grad():
        next_q_values_full = model(next_state_batch)
        next_q_values = torch.max(next_q_values_full, 1).values
    model.train()
    
    q_values_full = model(state_batch)
    q_values = torch.amax(q_values_full, 1)
    
    if torch.cuda.is_available():
        q_values = q_values.cuda()
    
    y_batch_list = tuple(torch.unsqueeze(reward + game_active_ind * gamma * next_q_value, 0) for reward, game_active_ind, next_q_value in
              zip(reward_tensor, game_active_tensor, next_q_values))
    
    y_batch_init = torch.cat(y_batch_list)
    y_batch = torch.reshape(y_batch_init, (-1,))
    
    
    if torch.cuda.is_available():
        y_batch = y_batch.cuda()

    optimizer.zero_grad()
    loss = criterion(q_values, y_batch)
    loss.backward()
    optimizer.step()
    
    current_action_count = env.action_count
    current_score = env.score
    current_tetrominoes = env.tetrominoes
    current_cleared_lines = env.cleared_lines

    print("    Episode: {}, Epoch: {}, Game ID: {}, Action Count: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
        episode,
        epoch,
        game_id,
        current_action_count,
        current_score,
        current_tetrominoes,
        current_cleared_lines))
    #writer.add_scalar('Train/Score', final_score, epoch - 1)
    #writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
    #writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

'''
Epsilon is the probability of making a RANDOM choice.
'''
def get_epsilon_for_episode(episode, opt):
    epsilon_range = opt.final_epsilon - opt.initial_epsilon
    episode_range = opt.num_decay_episodes - 1
    
    current_percent = (episode - 1) / episode_range
    # Clamp the percent to [0, 1]
    new_index = max(0, min(1, current_percent))
    
    epsilon = opt.initial_epsilon + current_percent * epsilon_range
    
    return epsilon 
    
    #epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
    #                    opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)

def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(747)
    else:
        torch.manual_seed(747)
    
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    
    if os.path.isdir(opt.saved_path):
        shutil.rmtree(opt.saved_path)
    os.makedirs(opt.saved_path)
    
    
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    action_names = env.get_action_names()
    model = DeepQNetworkAtari(4, len(action_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        model.cuda()

    replay_memory = ReplayMemory(frame_limit=opt.replay_memory_size)
    replay_memory_full = False
    epoch = 1
    game_id = 1
    episode = 1
    
    while episode <= opt.num_episodes:
        env.reset()
        print("Starting new Tetris game. Game ID: {}".format(game_id))
        is_game_over = False
        game_move_results = []
        epsilon = get_epsilon_for_episode(episode, opt)
        current_state = env.get_current_board_state()
        current_tensor = get_tensor_for_state(current_state)
        if torch.cuda.is_available():
            current_tensor = current_tensor.cuda()
         
        
        while not is_game_over:
            
            random_action = True
            if replay_memory_full:
                #epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                #        opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
                u = random()
                random_action = (u <= epsilon)
            
            action_index = 0
            
            if random_action:
                action_index = randrange(len(action_names))
            else:
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.unsqueeze(current_tensor, 0)
                    predictions = model(input_tensor)
                    action_index = torch.argmax(predictions)
                model.train()
            
            action_result_map = env.do_action_by_id(action_index)
            is_game_over = action_result_map["gameover"]
            reward = action_result_map["reward"]
            
            next_state = env.get_current_board_state()
            next_tensor = get_tensor_for_state(next_state)
            if torch.cuda.is_available():
                next_tensor = next_tensor.cuda()
            
            current_move_result = TetrisMoveResult(current_state, current_tensor, action_index, reward, is_game_over, next_state, next_tensor)
            game_move_results.append(current_move_result)
            
            current_state = next_state
            current_tensor = next_tensor
            
            if replay_memory_full:
                do_minibatch_update(episode, epoch, game_id, env, model, replay_memory, optimizer, criterion, opt.batch_size, opt.gamma)
                
                if epoch % opt.save_interval == 0:
                    torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))
                epoch += 1
                    
        replay_memory.insert(game_move_results, game_id)
        
        final_score = env.score
        final_tetrominoes = env.tetrominoes
        final_cleared_lines = env.cleared_lines
        
        if replay_memory_full:
            print("    Tetris game completed. Training Episode: {}, Game ID: {}, Score: {}, Tetrominoes: {}, Cleared Lines: {}, Epsilon: {}\n".format(
                episode,
                game_id,
                final_score,
                final_tetrominoes,
                final_cleared_lines,
                epsilon))
            episode += 1
        else:
            print("    Tetris setup game completed. Game ID: {}, Score: {}, Tetrominoes: {}, Cleared Lines: {}, Experience Replay Progress: {}/{}\n".format(
                game_id,
                final_score,
                final_tetrominoes,
                final_cleared_lines,
                replay_memory.get_size(),
                opt.replay_memory_size))
            replay_memory_full = replay_memory.is_full()
        
        
        
        game_id += 1
        
    torch.save(model, "{}/tetris".format(opt.saved_path))

if __name__ == "__main__":
    run_options = get_args()
    train(run_options)
