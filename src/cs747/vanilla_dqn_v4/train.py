"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
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

class TrainVanillaDqnV4(object):
   
    def __init__(self, run_options): 
        self.opt = run_options
    
    def initialize_torch_random(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(747)
        else:
            torch.manual_seed(747)
    
    def create_output_directories(self):
        if os.path.isdir(self.opt.log_path):
            shutil.rmtree(self.opt.log_path)
        os.makedirs(self.opt.log_path)
        
        if os.path.isdir(self.opt.saved_path):
            shutil.rmtree(self.opt.saved_path)
        os.makedirs(self.opt.saved_path)
    
    def add_to_cuda(self, *cuda_objects):
        if torch.cuda.is_available():
            for cuda_object in cuda_objects:
                cuda_object.cuda()
    
    '''
        Epsilon is the probability of taking a random action (explore).
        Otherwise, we will act according to the model output (exploit).
    '''
    def set_epsilon_for_episode(self):
        epsilon_range = self.opt.final_epsilon - self.opt.initial_epsilon
        episode_range = self.opt.num_decay_episodes - 1
        current_percent = (self.episode - 1) / episode_range
        
        # Clamp the percent to [0, 1]
        current_percent = max(0, min(1, current_percent))
        self.epsilon = self.opt.initial_epsilon + current_percent * epsilon_range
    
    def get_action_index_from_model(self, current_tensor):
        action_index = 0
        
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.unsqueeze(current_tensor, 0)
            predictions = self.model(input_tensor)
            action_index = torch.argmax(predictions)
        self.model.train()
        
        return action_index
    
    def print_game_complete(self):
        if self.replay_memory_full:
            print("    Tetris game completed. Training Episode: {}, Game ID: {}, Score: {}, Tetrominoes: {}, Cleared Lines: {}, Epsilon: {:.6f}\n"
                  .format(self.episode, self.game_id, self.env.score, self.env.tetrominoes, self.env.cleared_lines, self.epsilon))
        else:
            print("    Tetris setup game completed. Game ID: {}, Score: {}, Tetrominoes: {}, Cleared Lines: {}, Experience Replay Progress: {}/{}\n"
                  .format(self.game_id, self.env.score, self.env.tetrominoes, self.env.cleared_lines, self.replay_memory.get_size(), self.opt.replay_memory_size))
    
    def print_epoch_complete(self, random_action, action_index, loss_value):
        action_type = "EXPLORE" if random_action else "EXPLOIT"
        action_name = self.action_names[action_index]
        
        print("    Episode: {}, Epoch: {}, Game ID: {}, Action Count: {}, Loss: {:.6f}, Score: {}, Tetrominoes {}, Cleared lines: {}, Action Type: {}, Action Name: {}"
              .format(self.episode, self.epoch, self.game_id, self.env.action_count, loss_value, self.env.score, self.env.tetrominoes, self.env.cleared_lines, action_type, action_name))
        #writer.add_scalar('Train/Score', final_score, epoch - 1)
        #writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        #writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)    
    
    def get_tensor_for_state(self, board_state):
        return format_util.to_dqn_84x84(board_state)

    def do_minibatch_update(self):
        batch = self.replay_memory.get_random_sample(self.opt.batch_size)
        state_batch = torch.stack(tuple(sample.begin_tensor for sample in batch)).to(self.torch_device)
        next_state_batch = torch.stack(tuple(sample.next_tensor for sample in batch)).to(self.torch_device)
        
        #self.add_to_cuda(state_batch, next_state_batch)
        
        reward_list = [sample.reward for sample in batch]
        game_active_list = [(0 if sample.final_state_flag else 1) for sample in batch]
        reward_tensor = torch.tensor(reward_list).to(self.torch_device)
        game_active_tensor = torch.tensor(game_active_list).to(self.torch_device)
    
        #self.add_to_cuda(reward_tensor, game_active_tensor)
    
        self.model.eval()
        with torch.no_grad():
            next_q_values_full = self.model(next_state_batch).to(self.torch_device)
            next_q_values = torch.max(next_q_values_full, 1).to(self.torch_device).values
        self.model.train()
        
        q_values_full = self.model(state_batch).to(self.torch_device)
        q_values = torch.amax(q_values_full, 1).to(self.torch_device)
        
        #self.add_to_cuda(q_values)
        
        y_batch_list = tuple(torch.unsqueeze(reward + game_active_ind * self.opt.gamma * next_q_value, 0) for reward, game_active_ind, next_q_value in
                  zip(reward_tensor, game_active_tensor, next_q_values))
        
        y_batch_init = torch.cat(y_batch_list).to(self.torch_device)
        y_batch = torch.reshape(y_batch_init, (-1,)).to(self.torch_device)
        
        #self.add_to_cuda(y_batch)
    
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, y_batch)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
        
        
    
    def train(self):
        self.run_time = datetime.now()
        self.run_time_str = self.run_time.strftime("%b%d_%H%M%S")
        self.torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.initialize_torch_random()
        self.create_output_directories()
        self.env = Tetris()
        self.action_names = self.env.get_action_names()
        self.model = DeepQNetworkAtari(4, len(self.action_names)).to(self.torch_device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
        self.criterion = nn.MSELoss()
        self.replay_memory = ReplayMemory(frame_limit=self.opt.replay_memory_size)
        self.replay_memory_full = False
        self.epsilon = 1
        self.epoch = 1
        self.game_id = 1
        self.episode = 1
        

        
        #self.add_to_cuda(self.model)
        
        while self.episode <= self.opt.num_episodes:
            self.env.reset()
            print("Starting new Tetris game. Game ID: {}".format(self.game_id))
            is_game_over = False
            game_move_results = []
            self.set_epsilon_for_episode()
            current_state = self.env.get_current_board_state()
            current_tensor = self.get_tensor_for_state(current_state).to(self.torch_device)
            #self.add_to_cuda(current_tensor)
             
            while not is_game_over:
                random_action = True
                if self.replay_memory_full:
                    u = random()
                    random_action = (u <= self.epsilon)
                
                action_index = 0
                
                if random_action:
                    action_index = randrange(len(self.action_names))
                else:
                    action_index = self.get_action_index_from_model(current_tensor)
                
                action_result_map = self.env.do_action_by_id(action_index)
                is_game_over = action_result_map["gameover"]
                reward = action_result_map["reward"]
                
                next_state = self.env.get_current_board_state()
                next_tensor = self.get_tensor_for_state(next_state).to(self.torch_device)
                #self.add_to_cuda(next_tensor)
                                
                current_move_result = TetrisMoveResult(current_state, current_tensor, action_index, reward, is_game_over, next_state, next_tensor)
                game_move_results.append(current_move_result)
                
                current_state = next_state
                current_tensor = next_tensor
                
                if self.replay_memory_full:
                    loss_value = self.do_minibatch_update()
                    self.print_epoch_complete(random_action, action_index, loss_value)
                    
                    if self.episode % self.opt.save_interval == 0:
                        torch.save(self.model, "{}/{}_tetris_{}".format(self.opt.saved_path, self.run_time_str, self.episode))
                    self.epoch += 1
                        
            self.replay_memory.insert(game_move_results, self.game_id)
            self.print_game_complete()
            
            # If we are training, update the episode. Otherwise, check
            # if the replay buffer is full so we can begin training.
            if self.replay_memory_full:
                self.episode += 1
            else: 
                self.replay_memory_full = self.replay_memory.is_full()
            
            # Increment Game ID
            self.game_id += 1
            
        torch.save(self.model, "{}/{}_tetris".format(self.opt.saved_path, self.run_time_str))

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
    parser.add_argument("--save_interval", type=int, default=20) # This is a number of EPISODES
    parser.add_argument("--replay_memory_size", type=int, default=1000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print(__file__)
    run_options = get_args()
    trainDQN = TrainVanillaDqnV4(run_options)
    trainDQN.train()
