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

from cs747.models.deep_q_network import DeepQNetwork
from cs747.util.experience_replay import ReplayMemory
from cs747.sim.tetris import Tetris
from collections import deque
from cs747.sim.tetris_move_result import TetrisMoveResult



def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args

def choose_explore_or_exploit():
    return "EXPLORE"

def train(opt):
    '''
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    '''
    torch.manual_seed(123)
    
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    
    if os.path.isdir(opt.saved_path):
        shutil.rmtree(opt.saved_path)
    os.makedirs(opt.saved_path)
    
    
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    action_names = env.get_action_names()
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    #state = env.get_current_board_state()
    #tetris_board = env.get_current_board_state()
    
    '''
    board_tensor =  torch.tensor(tetris_board)
    print("Board Tensor below:")
    print(board_tensor)
    
    flat_tensor = torch.flatten(board_tensor)
    print("Flattened state below:")
    print(flat_tensor) 
    '''
    #if torch.cuda.is_available():
    #    model.cuda()
    #    state = state.cuda()

    #replay_memory = deque(maxlen=opt.replay_memory_size)
    replay_memory = ReplayMemory(frame_limit=opt.replay_memory_size)
    epoch = 0
    game_id = 1
    
    while epoch < opt.num_epochs:
        #next_steps = env.get_next_states()
        
        env.reset()
        is_game_over = False
        game_move_results = []
        current_state = env.get_current_board_state()
        
        while not is_game_over:
            # Exploration or exploitation
            epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                    opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
            u = random()
            random_action = u <= epsilon
            
            
            #next_actions, next_states = zip(*next_steps.items())
            #next_states = torch.stack(next_states)
            #if torch.cuda.is_available():
            #    next_states = next_states.cuda()
            
            action_index = 0
            
            if random_action:
                # This is the EXPLORE option
                #index = randint(0, len(next_steps) - 1)
                action_index = randrange(len(action_names))
            else:
                # This is the EXPLOIT option
                model.eval()
                with torch.no_grad():
                    board_tensor =  torch.tensor(current_state, requires_grad=False, dtype=torch.float)
                    board_tensor_flat = torch.flatten(board_tensor)
    
                    predictions = model(board_tensor_flat)
                    action_index = torch.argmax(predictions)
                model.train()
            
            
                #index = torch.argmax(predictions).item()
        
            action_result_map = env.do_action_by_id(action_index)
            is_game_over = action_result_map["gameover"]
            
            next_state = env.get_current_board_state()
            
            current_move_result = TetrisMoveResult(current_state, action_index, action_result_map["reward"], next_state)
            game_move_results.append(current_move_result)
            
        
        # Add a loop here to set the future discounted state for all actions
        # that we made during the game
        #
        # Note that the Q(s, a) value is: CurrentReward + DiscountedFutureRewards
        # So, we should add these values together when computing target values for the loss
        discounted_reward = 0
        
        for move_result in reversed(game_move_results):
            move_result.discounted_future_reward = discounted_reward
            discounted_reward = opt.gamma * (move_result.reward + discounted_reward)
        
        replay_memory.insert(game_move_results, game_id)
        
        final_score = env.score
        final_tetrominoes = env.tetrominoes
        final_cleared_lines = env.cleared_lines
        game_id += 1
        
        '''
        #if torch.cuda.is_available():
        #    next_state = next_state.cuda()
        
        replay_memory.append([state, reward, next_state, done])
        if done:

            state = env.reset()
            #if torch.cuda.is_available():
            #    state = state.cuda()
        else:
            state = next_state
            continue
        '''
        
        if replay_memory.get_size() < opt.replay_memory_size / 10:
            continue
        
        epoch += 1
        batch = replay_memory.get_random_sample(opt.batch_size)
        #batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        
        to_tensor = lambda x: torch.flatten(torch.tensor(x, dtype=torch.float))
        
        state_batch = torch.stack(tuple(to_tensor(sample.begin_state) for sample in batch))
        
        #state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        #state_batch = torch.stack(tuple(state for state in state_batch))
        
        y_list = [sample.reward + sample.discounted_future_reward for sample in batch]
        
        y_batch_init = torch.tensor(y_list)
        y_batch = torch.reshape(y_batch_init, (-1,))
        #reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        #next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        #if torch.cuda.is_available():
        #    state_batch = state_batch.cuda()
        #    reward_batch = reward_batch.cuda()
        #    next_state_batch = next_state_batch.cuda()

        q_values_full = model(state_batch)
        q_values = torch.amax(q_values_full, 1)
        #model.eval()
        #with torch.no_grad():
        #    next_prediction_batch = model(next_state_batch)
        #model.train()

        '''
        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]
        '''

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Game ID: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            game_id,
            final_score,
            final_tetrominoes,
            final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/tetris".format(opt.saved_path))


if __name__ == "__main__":
    run_options = get_args()
    train(run_options)
