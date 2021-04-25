"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
from random import random, randrange
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import time

from cs747.dqn_v6.dqn_model import DeepQNetworkAtariSmall
from cs747.dqn_v6.experience_replay import WeightedReplayMemory
from cs747.dqn_v6.tetris import Tetris
from cs747.dqn_v6.tetris_move_result import TetrisMoveResult


class TrainVanillaDqnV6(object):
    '''
    Class to train a NN for playing Tetris
    '''
   

    def __init__(self, run_options):
        '''
        Creates an instance to train the network, according to the
        parameters.
        ''' 
        self.opt = run_options
    

    def initialize_torch_random(self):
        '''
        Initialize PyTorch.
        '''
        if torch.cuda.is_available():
            torch.cuda.manual_seed(747)
        else:
            torch.manual_seed(747)


    def ensure_dir(self, dir_path):
        '''
        Create a directory if it does not already exist.
        '''
        directory = os.path.dirname(dir_path)
        if not os.path.exists(directory):
            os.makedirs(directory)


    def create_output_directories(self):
        '''
        Create a directory for output. This includes saved models and logs.
        '''
        dir_parts = []
        parent_path, file_name = os.path.split(__file__)
        dir_parts.insert(0, file_name)
        
        while file_name != "src":
            parent_path, file_name = os.path.split(parent_path)
            dir_parts.insert(0, file_name)
        
        version = "main"
        if(len(dir_parts) >= 3):
            version = dir_parts[2]
                
        run_time_dir = os.path.join(parent_path, "output", version, self.run_time_str)
        
        self.logs_directory = os.path.join(run_time_dir, "logs")
        if not os.path.exists(self.logs_directory):
            os.makedirs(self.logs_directory)
        
        self.models_directory = os.path.join(run_time_dir, "models")
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)

    def create_output_files(self):
        '''
        Initialize the output files to capture the results as we train the network.
        '''
        run_options_file_name = self.run_time_str + "_run_options.txt"
        self.run_options_file_path = os.path.join(self.logs_directory, run_options_file_name)
        self.run_options_file = open(self.run_options_file_path, "a")
        
        episodes_file_name = self.run_time_str + "_episodes.csv"
        self.episodes_file_path = os.path.join(self.logs_directory, episodes_file_name)
        self.episodes_file = open(self.epochs_file_path, "a")
        self.episodes_file_header_added = False
        
        epochs_file_name = self.run_time_str + "_epochs.csv"
        self.epochs_file_path = os.path.join(self.logs_directory, epochs_file_name)
        self.epochs_file = open(self.epochs_file_path, "a")
        
        games_file_name = self.run_time_str + "_games.txt"
        self.games_file_path = os.path.join(self.logs_directory, games_file_name)
        self.games_file = open(self.games_file_path, "a")
        self.games_file.write("This is the GAMES file.\n")
        self.games_file.write("Is there a new line here?\n")
        self.games_file.flush()
    
    def write_options_file(self):
        '''
        Writes the options file and closes the file.
        This method should ony be called once, as it closed the file!
        '''
        self.run_options_file.write("Run Key is: " + self.run_time_str + "\n")
        self.run_options_file.write("Training File is: " + str(__file__) + "\n")
        self.run_options_file.write("\nArguments from command line are below:" + "\n")
        for k, v in vars(run_options).items():
            self.run_options_file.write(f"{k} = {v}\n")
        
        self.run_options_file.close()
    
    
    def close_output_files(self):
        '''
        Close and flush all of the output files. 
        '''
        if not self.games_file.closed:
            self.games_file.close()
        
        if not self.epochs_file.closed:
            self.epochs_file.close()
            
        if not self.episodes_file.closed:
            self.episodes_file.close()
        
        if not self.run_options_file.closed:
            self.run_options_file.close()

 
    def get_current_time_ms(self):
        '''
        Get the current time. We use this to time how long each game takes.
        '''           
        return round(time.time() * 1000)


    def set_epsilon_for_episode(self):
        '''
        Epsilon is the probability of taking a random action (explore).
        Otherwise, we will act according to the model output (exploit).
        '''        
        epsilon_range = self.opt.final_epsilon - self.opt.initial_epsilon
        episode_range = self.opt.num_decay_episodes - 1
        current_percent = (self.episode - 1) / episode_range
        
        # Clamp the percent to [0, 1]
        current_percent = max(0, min(1, current_percent))
        self.epsilon = self.opt.initial_epsilon + current_percent * epsilon_range
    

    @torch.no_grad
    def get_best_action_from_model(self, current_tensor):
        '''
        Choose an action by passing the current state to the model and
        choosing the one with the highest predicted Q-value.
        '''
        input_tensor = torch.unsqueeze(current_tensor, 0).to(self.torch_device)
        predictions = self.model(input_tensor)
        print("Predictions = " + str(predictions))
        print(predictions.size())
        max_elements, max_idxs = torch.max(predictions, dim=1)
        
        return max_elements, max_idxs
    
    def create_episode_properties(self):
        '''
        Create the list of parameters that will be used to summarize the performance for the episode. 
        '''
        self.episode_property_names = ["Replay_Memory_Full", "Episode_Num", "Total_Episodes", "Game_ID", "Reward_Sum", "Tetrominoes", 
                                       "Action_Count", "Cleared_Lines", "Duration", "Epsilon", "Replay_Memory_Size", "Replay_Memory_Capacity"]
    
    
    def write_episodes_file(self):
        '''
        Collect the data needed to log the performance for each episode. 
        '''
        episode_property_values = {}
        episode_property_values["Replay_Memory_Full"] = str(self.replay_memory_full)
        episode_property_values["Episode_Num"] = str(self.episode)
        episode_property_values["Total_Episodes"] = str(self.opt.num_episodes)
        episode_property_values["Game_ID"] = str(self.game_id)
        episode_property_values["Reward_Sum"] = str(self.env.discounted_reward)
        episode_property_values["Tetrominoes"] = str(self.env.tetrominoes)
        episode_property_values["Action_Count"] = str(self.env.action_count)
        episode_property_values["Cleared_Lines"] = str(self.env.cleared_lines)
        episode_property_values["Duration"] = "{:.2f}s".format(self.game_time_ms / 1000)
        episode_property_values["Epsilon"] = str(self.epsilon)
        episode_property_values["Replay_Memory_Size"] = str(self.replay_memory.get_size())
        episode_property_values["Replay_Memory_Capacity"] = str(self.opt.replay_memory_size)
        
        if not self.episodes_file_header_added:
            header_row = ",".join(episode_property_values.keys())
            self.episodes_file.write(header_row + "\n")
            self.episodes_file_header_added = True
        
        data_row = ",".join(episode_property_values.values())
        self.episodes_file.write(data_row + "\n")
        
        # Flush the file so the output gets written in case the program is terminated.
        self.episodes_file.flush()
        
    
    def output_episode_complete(self):
        '''
        
        '''
        for 
        
    
    def print_episode_complete(self):
        if self.replay_memory_full:
            print("    Training Episode: {}/{}, Game ID: {}, Reward Sum: {:.2f}, Tetrominoes: {}, Cleared Lines: {}, Duration: {:.2f}s, Epsilon: {:.4f}\n"
                  .format(self.episode, self.opt.num_episodes, self.game_id, self.env.discounted_reward, self.env.tetrominoes, self.env.cleared_lines, self.game_time_ms / 1000, self.epsilon))
        else:
            print("    Setup Episode: Game ID: {}, Reward Sum: {}, Tetrominoes: {}, Cleared Lines: {}, Duration: {:.3f}s, Experience Replay Progress: {}/{}\n"
                  .format(self.game_id, self.env.discounted_reward, self.env.tetrominoes, self.env.cleared_lines, self.game_time_ms / 1000, self.replay_memory.get_size(), self.opt.replay_memory_size))
    
    def print_epoch_complete(self, random_action, action_index, loss_value, reward):
        action_type = "EXPLORE" if random_action else "EXPLOIT"
        action_name = self.action_names[action_index]
        
        print("    Episode: {}, Epoch: {}, Game ID: {}, Action Count: {}, Loss: {:.6f}, Reward: {}, Tetrominoes {}, Cleared lines: {}, Action Type: {}, Action Name: {}"
              .format(self.episode, self.epoch, self.game_id, self.env.action_count, loss_value, reward, self.env.tetrominoes, self.env.cleared_lines, action_type, action_name))
    
    def get_tensor_for_state(self, board_state):
        board_tensor = torch.from_numpy(board_state).to(self.torch_device)
        state_tensor = torch.unsqueeze(board_tensor, 0).to(self.torch_device)
        return state_tensor

    def do_minibatch_update(self):
        batch = self.replay_memory.get_random_weighted_sample(self.opt.batch_size)
        state_batch = torch.stack(tuple(sample.begin_tensor for sample in batch)).to(self.torch_device)
        next_state_batch = torch.stack(tuple(sample.next_tensor for sample in batch)).to(self.torch_device)
        
        reward_list = [sample.reward for sample in batch]
        game_active_list = [(0 if sample.final_state_flag else 1) for sample in batch]
        reward_tensor = torch.tensor(reward_list).to(self.torch_device)
        game_active_tensor = torch.tensor(game_active_list).to(self.torch_device)
    
        self.model.eval()
        with torch.no_grad():
            next_q_values_full = self.model(next_state_batch).to(self.torch_device)
            next_q_values = torch.max(next_q_values_full, 1).values
        self.model.train()
        
        q_values_full = self.model(state_batch).to(self.torch_device)
        q_values = torch.amax(q_values_full, 1).to(self.torch_device)
        
        y_batch_list = tuple(torch.unsqueeze(reward + game_active_ind * self.opt.gamma * next_q_value, 0) for reward, game_active_ind, next_q_value in
                  zip(reward_tensor, game_active_tensor, next_q_values))
        
        y_batch_init = torch.cat(y_batch_list).to(self.torch_device)
        y_batch = torch.reshape(y_batch_init, (-1,)).to(self.torch_device)
    
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, y_batch)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def print_run_config(self):
        print("=================================================")
        print("Starting Training for Tetris with Deep-Q-Networks")
        print("Run Key is: " + self.run_time_str)
        print("Training File is: " + str(__file__))
        print("\nArguments from command line are below:")
        for k, v in vars(run_options).items():
            print(f"{k} = {v}")
        
        print("===============================================\n")
    
    def initialize_training(self):
        self.run_time = datetime.now()
        self.run_time_str = self.run_time.strftime("%b%d_%H%M%S")
        
        self.print_run_config()
        
        self.torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.initialize_torch_random()
        self.create_output_directories()
        self.create_output_files()
        
        self.env = Tetris(height=self.opt.board_height, width=self.opt.board_width, block_size=self.opt.block_size, gamma=self.opt.gamma)
        self.action_names = self.env.get_action_names()
        self.model = DeepQNetworkAtariSmall(len(self.action_names)).to(self.torch_device)
        self.target_network = DeepQNetworkAtariSmall(len(self.action_names)).to(self.torch_device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate)
        self.criterion = nn.MSELoss()
        self.replay_memory = WeightedReplayMemory(capacity=self.opt.replay_memory_size)
        self.replay_memory_full = False
        self.game_time_ms = 0
        self.epsilon = 1
        self.epoch = 1
        self.game_id = 1
        self.episode = 1
    
    @torch.no_grad()    
    def update_target_network(self):
        '''
        Perform a momentum update for the weights of the target network.
        '''
        for param_q_network, param_target_network in zip(self.model.parameters(), self.target_network.parameters()):
            param_target_network.data = param_target_network.data * self.opt.target_network_momentum + param_q_network.data * (1. - self.target_network_momentum)

    
    def train(self):
        '''
        Main training loop for the model
        '''
        
        # Set up all the variables that we need for training
        self.initialize_training()
        
        while self.episode <= self.opt.num_episodes:
            self.env.reset()
            print("Starting new Tetris game. Game ID: {}".format(self.game_id))
            is_game_over = False
            #game_move_results = []
            self.set_epsilon_for_episode()
            
            temp_current_state = np.array(self.env.get_current_board_state(), dtype=np.float32)
            current_state = np.clip(temp_current_state, -1, 1)
            current_tensor = self.get_tensor_for_state(current_state).to(self.torch_device)
            start_time_ms = self.get_current_time_ms()
             
            while not is_game_over:
                random_action = True
                if self.replay_memory_full:
                    u = random()
                    random_action = (u <= self.epsilon)
                
                action_index = 0
                
                model_value, model_index = self.get_best_action_from_model(current_tensor)
                print("model_index = " + str(model_index))
                print("model_value = " + str(model_value))
                
                if random_action:
                    action_index = randrange(len(self.action_names))
                else:
                    action_index = model_index
                               
                
                action_result_map = self.env.do_action_by_id(action_index)
                is_game_over = action_result_map["gameover"]
                reward = action_result_map["reward"]
                
                temp_next_state = np.array(self.env.get_current_board_state(), dtype=np.float32)
                next_state = np.clip(temp_next_state, -1, 1)
                next_tensor = self.get_tensor_for_state(next_state).to(self.torch_device)
                                
                current_move_result = TetrisMoveResult(current_state, current_tensor, action_index, reward, is_game_over, next_state, next_tensor)
                self.replay_memory.insert(current_move_result, abs(reward) + 0.01)
                #game_move_results.append(current_move_result)
                
                current_state = next_state
                current_tensor = next_tensor
                
                if self.replay_memory_full:
                    loss_value = 0
                    if self.epoch % 100 == 0:
                        self.update_target_network()
                    
                    if self.epoch % 5 == 0:
                        loss_value = self.do_minibatch_update()
                    
                    if self.episode % 50 == 0:
                        self.print_epoch_complete(random_action, action_index, loss_value, reward)
                    
                    if self.episode % self.opt.save_interval == 0:
                        model_path = os.path.join(self.models_directory, "tetris_{}.pt".format(self.episode))
                        torch.save(self.model, model_path)
                    
                    self.epoch += 1
                        
            #self.replay_memory.insert(game_move_results, self.game_id)
            end_time_ms = self.get_current_time_ms()
            self.game_time_ms = end_time_ms - start_time_ms
            self.print_game_complete()
            
            # If we are training, update the episode. Otherwise, check
            # if the replay buffer is full so we can begin training.
            if self.replay_memory_full:
                self.episode += 1
            else: 
                self.replay_memory_full = self.replay_memory.is_full()
            
            # Increment Game ID
            self.game_id += 1
        
        model_path = os.path.join(self.models_directory, "tetris_final.pt")    
        torch.save(self.model, model_path)
        
        self.close_output_files()

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--board_width", type=int, default=10, help="The tetris board width")
    parser.add_argument("--board_height", type=int, default=20, help="The tetris board height")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of samples per batch")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=.75)
    parser.add_argument("--final_epsilon", type=float, default=0.001)
    #parser.add_argument("--num_decay_epochs", type=float, default=2000)
    #parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--num_decay_episodes", type=int, default=25000)
    parser.add_argument("--num_episodes", type=int, default=30000)
    parser.add_argument("--pre_trained_model", type=string, default=None)
    parser.add_argument("--save_interval", type=int, default=500)  # This is a number of EPISODES
    parser.add_argument("--replay_memory_size", type=int, default=16384,
                        help="Number of epochs between testing phases")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    run_options = get_args()
    trainDQN = TrainVanillaDqnV6(run_options)
    trainDQN.train()
