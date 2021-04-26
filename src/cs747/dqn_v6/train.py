"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
from random import random, randrange
from datetime import datetime
import numpy as np
import copy

import torch
import torch.nn as nn
import time

from cs747.dqn_v6.dqn_model import DeepQNetworkAtariSmall
from cs747.dqn_v6.experience_replay import WeightedReplayMemory
from cs747.dqn_v6.tetris import Tetris
from cs747.dqn_v6.tetris_move_result import TetrisMoveResult


class TrainVanillaDqnV6():
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
        
        self.output_directory = os.path.join(parent_path, "output")        
        run_time_dir = os.path.join(self.output_directory, version, self.run_time_str)
        
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
        self.episodes_file = open(self.episodes_file_path, "a")
        self.episodes_file_header_added = False
        
        episodes_log_file_name = self.run_time_str + "_episodes_log.txt"
        self.episodes_log_file_path = os.path.join(self.logs_directory, episodes_log_file_name)
        self.episodes_log_file = open(self.episodes_log_file_path, "a")
        
        epochs_file_name = self.run_time_str + "_epochs.csv"
        self.epochs_file_path = os.path.join(self.logs_directory, epochs_file_name)
        self.epochs_file = open(self.epochs_file_path, "a")
        
        epochs_log_file_name = self.run_time_str + "_epochs_log.txt"
        self.epochs_log_file_path = os.path.join(self.logs_directory, epochs_log_file_name)
        self.epochs_log_file = open(self.epochs_log_file_path, "a")
        
        games_file_name = self.run_time_str + "_games.txt"
        self.games_file_path = os.path.join(self.logs_directory, games_file_name)
        self.games_file = open(self.games_file_path, "a")
        self.games_file.write("This is the GAMES file.\n")
        self.games_file.write("Is there a new line here?\n")
        self.games_file.flush()
    
    def write_run_options_file(self):
        '''
        Writes the options file and closes the file.
        This method should ony be called once, as it closed the file!
        '''
        self.run_options_file.write("Run Key is: " + self.run_time_str + "\n")
        self.run_options_file.write("Training File is: " + str(__file__) + "\n")
        self.run_options_file.write("\nArguments from command line are below:" + "\n")
        for k, v in vars(self.opt).items():
            self.run_options_file.write(f"{k} = {v}\n")
        
        self.run_options_file.close()
    
    
    def close_output_files(self):
        '''
        Close and flush all of the output files. 
        '''
        if not self.games_file.closed:
            self.games_file.close()
            
        if not self.epochs_log_file.closed:
            self.epochs_log_file.close()
        
        if not self.epochs_file.closed:
            self.epochs_file.close()
            
        if not self.episodes_log_file.closed:
            self.episodes_log_file.close()
            
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
        if self.replay_memory_full:
            epsilon_range = self.opt.final_epsilon - self.opt.initial_epsilon
            episode_range = self.opt.num_decay_episodes - 1
            current_percent = (self.episode - 1) / episode_range
            
            # Clamp the percent to [0, 1]
            current_percent = max(0, min(1, current_percent))
            self.epsilon = self.opt.initial_epsilon + current_percent * epsilon_range            
        else:
            # Check if we wanted to specify a different epsilon when loading the replay memory
            # This could be used if we want to continue training a model and not have
            # random replay experience.
            if self.opt.replay_memory_init_epsilon >= 0:
                self.epsilon = self.opt.replay_memory_init_epsilon
            else:
                self.epsilon = self.opt.initial_epsilon

    @torch.no_grad()
    def get_q_values(self, eval_model, state_tensor):
        model_input = torch.unsqueeze(state_tensor, 0).to(self.torch_device)
        model_output = eval_model(model_input)
        predictions = torch.squeeze(model_output, dim=0).to(self.torch_device)
        return predictions


    @torch.no_grad()
    def get_best_action_from_model(self, current_tensor):
        '''
        Choose an action by passing the current state to the model and
        choosing the one with the highest predicted Q-value.
        '''
        input_tensor = torch.unsqueeze(current_tensor, 0).to(self.torch_device)
        predictions = self.model(input_tensor)
        #print("Predictions = " + str(predictions))
        #print(predictions.size())
        max_elements, max_idxs = torch.max(predictions, dim=1)
        
        return max_elements, max_idxs
    
    
    def write_episodes_csv_file(self):
        '''
        Collect the data needed to log the performance for each episode. 
        '''
        episode_property_values = {}
        episode_property_values["Replay_Memory_Full"] = str(self.replay_memory_full)
        episode_property_values["Episode_Num"] = str(self.episode)
        episode_property_values["Total_Episodes"] = str(self.opt.num_episodes)
        episode_property_values["Game_ID"] = str(self.game_id)
        episode_property_values["Reward_Sum"] = str(self.episode_q_value_sum)
        episode_property_values["Tetrominoes"] = str(self.env.tetrominoes)
        episode_property_values["Action_Count"] = str(self.env.action_count)
        episode_property_values["Cleared_Lines"] = str(self.env.cleared_lines)
        episode_property_values["Duration"] = "{:.2f}".format(self.game_time_ms / 1000)
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
    
    def write_epochs_csv_file(self):
        '''
        Collect the data needed to log a single epoch. 
        '''
        pass
        
        '''
        epoch_property_values = {}
        epoch_property_values["Replay_Memory_Full"] = str(self.replay_memory_full)
        epoch_property_values["Episode_Num"] = str(self.episode)
        epoch_property_values["Total_Episodes"] = str(self.opt.num_episodes)
        epoch_property_values["Game_ID"] = str(self.game_id)
        epoch_property_values["Reward_Sum"] = str(self.env.discounted_reward)
        epoch_property_values["Tetrominoes"] = str(self.env.tetrominoes)
        epoch_property_values["Action_Count"] = str(self.env.action_count)
        epoch_property_values["Cleared_Lines"] = str(self.env.cleared_lines)
        epoch_property_values["Duration"] = "{:.2f}s".format(self.game_time_ms / 1000)
        epoch_property_values["Epsilon"] = str(self.epsilon)
        epoch_property_values["Replay_Memory_Size"] = str(self.replay_memory.get_size())
        epoch_property_values["Replay_Memory_Capacity"] = str(self.opt.replay_memory_size)
        '''
        # Flush the file so the output gets written in case the program is terminated.

    
    def write_to_file(self, file_obj, message):
        '''
        Writes a string to a file
        '''
        file_obj.write(message + "\n")
        file_obj.flush()

    
    def get_episode_info_message(self):
        '''
        Get the info message for the episode.
        '''
        if self.replay_memory_full:
            info_message = "Training Game. Episode: {}/{}, Run ID: {}, Game ID: {}, Actions: {}, Reward Sum: {:.2f}, Tetrominoes: {}, Cleared Lines: {}, Duration: {:.3f}s, Epsilon: {:.5f}".format(
                self.episode, self.opt.num_episodes, self.run_time_str, self.game_id, self.env.action_count, self.episode_q_value_sum, self.env.tetrominoes, 
                self.env.cleared_lines, self.game_time_ms / 1000, self.epsilon)
        else:
            info_message = "Setup Game. Run ID: {}, Game ID: {}, Actions: {}, Reward Sum: {:.2f}, Tetrominoes: {}, Cleared Lines: {}, Duration: {:.3f}s, Experience Replay Progress: {}/{}".format(
                self.run_time_str, self.game_id, self.env.action_count, self.episode_q_value_sum, self.env.tetrominoes, 
                self.env.cleared_lines, self.game_time_ms / 1000, self.replay_memory.get_size(), self.opt.replay_memory_size)
        
        return info_message

    
    def get_epoch_info_message(self):
        '''
        Print the summary for the current epoch.
        '''
        action_type = "EXPLORE" if self.epoch_random_action_flag else "EXPLOIT"
        
        return "Epoch: {}, Episode: {}/{}, Game ID: {}, Epsilon: {:.5f}, Action Count: {}, Loss: {:.6f}, Reward: {}, Tetrominoes {}, Cleared lines: {}, Action Type: {}, Action Name: {}".format(
            self.epoch, self.episode, self.opt.num_episodes, self.game_id, self.epsilon, self.env.action_count, self.minibatch_update_loss, 
            self.epoch_reward, self.env.tetrominoes, self.env.cleared_lines, action_type, self.epoch_action_name)

    
    def get_tensor_for_state(self, board_state):
        '''
        Convert the current board_state into a tensor for PyTorch
        '''
        numpy_arr = np.array(board_state, dtype=np.float32)
        clipped_numpy_arr = np.clip(numpy_arr, -1, 1)
        board_tensor = torch.from_numpy(clipped_numpy_arr).to(self.torch_device)
        state_tensor = torch.unsqueeze(board_tensor, 0).to(self.torch_device)
        return state_tensor

    def do_minibatch_update(self):
        '''
        Perform a mini-batch update on the Q-Network weights using samples from the experience replay memory. 
        '''
        batch = self.replay_memory.get_random_weighted_sample(self.opt.minibatch_size)
        state_batch = torch.stack(tuple(sample.begin_tensor for sample in batch)).to(self.torch_device)
        state_batch.requires_grad = False
        
        next_state_batch = torch.stack(tuple(sample.next_tensor for sample in batch)).to(self.torch_device)
        next_state_batch.requires_grad = False
        
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
        #q_values.requires_grad = True
        
        y_batch_list = tuple(torch.unsqueeze(reward + game_active_ind * self.opt.gamma * next_q_value, 0) for reward, game_active_ind, next_q_value in
                  zip(reward_tensor, game_active_tensor, next_q_values))
        
        y_batch_init = torch.cat(y_batch_list).to(self.torch_device)
        y_batch = torch.reshape(y_batch_init, (-1,)).to(self.torch_device)
        #y_batch.requires_grad = True
    
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, y_batch)
        loss.backward()
        self.optimizer.step()
        
        self.minibatch_update_loss = loss.item()
    
    def print_run_config(self):
        '''
        Print the run configuration used for this training run.
        '''
        print("=================================================")
        print("Starting Training for Tetris with Deep-Q-Networks")
        print("Run Key is: " + self.run_time_str)
        print("Training File is: " + str(__file__))
        print("\nArguments from command line are below:")
        for k, v in vars(self.opt).items():
            print(f"{k} = {v}")
        
        print("===============================================\n")
    
    def save_model(self, model_filename):
        model_path = os.path.join(self.models_directory, model_filename + ".tar")
        model_save_params = {
                                'epoch': self.epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': self.minibatch_update_loss,
                                'episode': self.episode,
                                'game_id': self.game_id,
                                'step_id': self.step_id
                            }
        
        torch.save(model_save_params, model_path)
    
    def initialize_model(self):
        source_model_path = self.opt.source_model_path
        
        if source_model_path is not None:
            if os.path.isabs(source_model_path):
                load_path = source_model_path
            else:
                load_path = os.path.join(self.output_directory, self.opt.source_model_path)
            
            '''
            checkpoint = torch.load(load_path)
            self.model = DeepQNetworkAtariSmall(len(self.action_names))
            self.model.load_state_dict()
            '''
            self.model = torch.load(load_path)
            self.model.to(self.torch_device)
            self.model.train()
        
        else:
            self.model = DeepQNetworkAtariSmall(len(self.action_names)).to(self.torch_device)
            self.model.train()
        
        self.target_network = DeepQNetworkAtariSmall(len(self.action_names)).to(self.torch_device)
        self.target_network.load_state_dict(copy.deepcopy(self.model.state_dict()))
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate)
        self.criterion = nn.MSELoss()
        
    
    def initialize_training(self):
        '''
        Set up all variables that are needed for training run.
        '''
        self.run_time = datetime.now()
        self.run_time_str = self.run_time.strftime("%b%d_%H%M%S")
        
        self.print_run_config()
        
        self.torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.initialize_torch_random()
        self.create_output_directories()
        self.create_output_files()
        
        self.write_run_options_file()
        
        self.env = Tetris(height=self.opt.board_height, width=self.opt.board_width, block_size=self.opt.block_size, gamma=self.opt.gamma)
        self.action_names = self.env.get_action_names()
        
        self.initialize_model()

        self.replay_memory = WeightedReplayMemory(capacity=self.opt.replay_memory_size)
        self.replay_memory_full = False
        self.game_time_ms = 0
        self.epsilon = 1
        
        # Global Game Count
        self.game_id = 1
        # Count of only Training Games
        self.epoch = 0
        
        # Global Action Count
        self.step_id = 1
        # Count of Actions on Training Games
        self.episode = 0
    
    @torch.no_grad()    
    def update_target_network(self):
        '''
        Perform a momentum update for the weights of the target network.
        '''
        for param_q_network, param_target_network in zip(self.model.parameters(), self.target_network.parameters()):
            target_network_weight = param_target_network.data * self.opt.target_network_momentum
            q_network_weight = param_q_network.data * (1. - self.opt.target_network_momentum)
            param_target_network.data = target_network_weight + q_network_weight
    
    @torch.no_grad()
    def refresh_replay_memory(self):
        all_move_results = self.replay_memory.object_buffer
        
        begin_tensor_list = [move_result.begin_tensor for move_result in all_move_results]
        begin_tensor_full = torch.stack(begin_tensor_list)
        
        next_tensor_list = [move_result.next_tensor for move_result in all_move_results]
        next_tensor_full = torch.stack(next_tensor_list)
        
        index_action_tensor = torch.tensor([move_result.action for move_result in all_move_results])
        reward_tensor = torch.tensor([move_result.reward for move_result in all_move_results])          
        index_tensor = torch.tensor(range(len(all_move_results)))
        
        current_q_values_full = self.model(begin_tensor_full)
        current_q_values = current_q_values_full[index_tensor, index_action_tensor]
        
        next_q_values_full = self.target_network(next_tensor_full)
        next_q_values = torch.max(next_q_values_full, dim=1).values
        
        for current_q_value, next_q_value, reward, move_result in zip(current_q_values, next_q_values, reward_tensor, all_move_results):
            move_result.begin_q_value = current_q_value.item()
            move_result.next_q_value = next_q_value.item()
            estimate_error = current_q_value - (reward + self.opt.gamma * next_q_value)
            move_result.weight = abs(estimate_error.item())
        
        self.replay_memory.reset_weights()

    
    def step_finished(self):
        '''
        Function to handle all processing that occurs when a STEP is completed
        '''
        
        if self.replay_memory_full:
            # Do a mini-batch
            if self.epoch % self.opt.minibatch_update_epoch_freq == 0:
                self.do_minibatch_update()
            
            # Target Network Update
            if self.epoch % self.opt.target_network_update_epoch_freq == 0:
                self.update_target_network()
                self.refresh_replay_memory()
            
            # Logging / Printing
            print_flag = self.epoch % self.opt.print_epoch_freq == 0
            log_flag = self.epoch % self.opt.log_file_epoch_freq == 0
            
            if print_flag or log_flag:
                epoch_info_message = self.get_epoch_info_message()
            
                if print_flag:
                    print("    " + epoch_info_message)
                
                if log_flag:
                    self.write_to_file(self.epochs_log_file, epoch_info_message)
            
            # CSV File Output
            csv_flag = self.epoch % self.opt.log_csv_epoch_freq == 0
            
            if csv_flag:
                self.write_to_file(self.epochs_file, "Epoch = " +  str(self.epoch))
            
            self.epoch += 1
        
        # Update the step
        self.step_id += 1
    
    
    def is_episode_event_active(self, episode_freq):
        '''
        Determine whether an episode logging / writing event should be performed.
        '''
        is_active = False
        
        if episode_freq == 0:
            is_active = True
        else:
            if episode_freq > 0 and self.episode % episode_freq == 0:
                is_active = True
        
        return is_active
                
    
    def game_finished(self):
        '''
        Function to handle all processing that occurs when a GAME is completed
        '''
        
        # Handle Print / Logging
        print_flag = self.is_episode_event_active(self.opt.print_episode_freq)
        log_file_flag = self.is_episode_event_active(self.opt.log_file_episode_freq)
        
        if print_flag or log_file_flag:
            episode_info_message = self.get_episode_info_message()
            
            if print_flag:
                print(episode_info_message)
        
            if log_file_flag:
                self.write_to_file(self.episodes_log_file, episode_info_message)
        
        # Handle CSV Output
        csv_file_flag = self.is_episode_event_active(self.opt.log_csv_episode_freq)
        
        if csv_file_flag:
            self.write_episodes_csv_file()
            
        
        # Handle Model Saviing
        if self.episode > 0 and self.episode % self.opt.save_model_episode_freq == 0:
            model_filename = "tetris_{}_{}".format(self.episode, self.epoch)
            self.save_model(model_filename)
            
        # If we are training, update the episode. Otherwise, check
        # if the replay buffer is full so we can begin training.
        if not self.replay_memory_full:
            self.replay_memory_full = self.replay_memory.is_full()
        
        if self.replay_memory_full:
            if self.episode == 0 and self.epoch == 0:
                # Only update this when we've completed filling the replay buffer.
                self.epoch = 1
                
            self.episode += 1
        
        # Increment Game ID
        self.game_id += 1
    
    def train(self):
        '''
        Main training loop for the model
        '''
        
        # Set up all the variables that we need for training
        self.initialize_training()
        
        while self.episode <= self.opt.num_episodes:
            # Tell the environment to start a new game
            self.env.reset()
            
            #print("Starting new Tetris game. Game ID: {}".format(self.game_id))
            self.epoch_game_over = False
            game_move_results = []
            self.set_epsilon_for_episode()
            
            current_state = self.env.get_current_board_state()
            current_tensor = self.get_tensor_for_state(current_state).to(self.torch_device)
            
            start_time_ms = self.get_current_time_ms()
            
            while not self.epoch_game_over:
                self.epoch_random_action_flag = True
                self.minibatch_update_loss = -1
                
                if self.replay_memory_full:
                    u = random()
                    self.epoch_random_action_flag = (u <= self.epsilon)
                
                current_q_values = self.get_q_values(self.model, current_tensor)
                
                if self.epoch_random_action_flag:
                    self.epoch_action_index = randrange(len(self.action_names))
                    self.epoch_q_value = current_q_values[self.epoch_action_index].item()
                else:
                    model_value, model_index = torch.max(current_q_values, dim=0)
                    self.epoch_action_index = model_index.item()
                    self.epoch_q_value = model_value.item()
                
                action_result_map = self.env.do_action_by_id(self.epoch_action_index)
                self.epoch_action_name = self.action_names[self.epoch_action_index]
                self.epoch_game_over = action_result_map["gameover"]
                self.epoch_reward = action_result_map["reward"]
                
                next_state = self.env.get_current_board_state()
                next_tensor = self.get_tensor_for_state(next_state).to(self.torch_device)
                
                # Use the target network estimate: r + gamma * max:a_next [Q_target(s_next, a_next)] 
                # Compare to the DQN estimate for Q(s, a)   
                next_q_values = self.get_q_values(self.target_network, next_tensor)
                self.epoch_target_q_value = torch.max(next_q_values).item()
                self.epoch_target_total = self.epoch_reward + self.opt.gamma * self.epoch_target_q_value
                self.q_value_error = self.epoch_q_value - self.epoch_target_total
                self.epoch_weight = abs(self.q_value_error)
                                
                current_move_result = TetrisMoveResult(current_state, current_tensor, self.epoch_q_value, self.epoch_action_index, 
                                                       self.epoch_action_name, self.epoch_reward, self.epoch_game_over, next_state, 
                                                       next_tensor, self.epoch_target_q_value, self.epoch_weight, self.epoch)
                
                self.replay_memory.insert(current_move_result)
                game_move_results.append(current_move_result)
                
                current_state = next_state
                current_tensor = next_tensor
                
                # Do the processing for the end of the STEP.
                self.step_finished()
                        
            #self.replay_memory.insert(game_move_results, self.game_id)
            end_time_ms = self.get_current_time_ms()
            self.game_time_ms = end_time_ms - start_time_ms
            
            # Calculate the total discounted q-value for the game
            q_value_sum = 0
            for move_result in reversed(game_move_results):
                q_value_sum = move_result.reward + self.opt.gamma * q_value_sum
                move_result.true_q_value = q_value_sum
            
            self.episode_q_value_sum = q_value_sum
                
            
            # Do the processing for the end of the GAME.
            self.game_finished()
            
        # Save the final model
        model_filename = "tetris_final_{}_{}".format(self.episode - 1, self.epoch - 1)
        self.save_model(model_filename)
        
        # Clean up
        self.close_output_files()


def get_args():
    '''
    Parse the arguments to the training run.
    '''
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")
    
    # Tetris Parameters
    parser.add_argument("--board_width", type=int, default=10, help="The tetris board width")
    parser.add_argument("--board_height", type=int, default=20, help="The tetris board height")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")

    # Parameters for RL    
    parser.add_argument("--source_model_path", type=str, default=None, help="Location of the saved model to load for training (relative to output/ or absolute).")
    parser.add_argument("--replay_memory_size", type=int, default=16384, help="Number of actions stored in the experience replay memory")
    parser.add_argument("--minibatch_size", type=int, default=512, help="The number of samples per batch")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target_network_momentum", type=float, default=0.999)

    # EPISODE based events
    parser.add_argument("--print_episode_freq", type=int, default=0, help="Negative for Never. Zero for always (and setup). Positive for episode multiples")
    parser.add_argument("--log_file_episode_freq", type=int, default=0, help="Negative for Never. Zero for always (and setup). Positive for episode multiples")
    parser.add_argument("--log_csv_episode_freq", type=int, default=1, help="Negative for Never. Zero for always (and setup). Positive for episode multiples")
    parser.add_argument("--save_model_episode_freq", type=int, default=2000)
    
    parser.add_argument("--num_episodes", type=int, default=100000)
    parser.add_argument("--num_decay_episodes", type=int, default=80000)
    
    parser.add_argument("--replay_memory_init_epsilon", type=float, default=-1.0, help="Epsilon to use while populating replay memory. Use -1 to ignore and use initial_epsilon")
    parser.add_argument("--initial_epsilon", type=float, default=.75)
    parser.add_argument("--final_epsilon", type=float, default=0.001)
    
    # EPOCH based events
    parser.add_argument("--print_epoch_freq", type=int, default=40)
    parser.add_argument("--log_file_epoch_freq", type=int, default=40)
    parser.add_argument("--log_csv_epoch_freq", type=int, default=40)
    
    parser.add_argument("--target_network_update_epoch_freq", type=int, default=12000)
    parser.add_argument("--minibatch_update_epoch_freq", type=int, default=4)
    
        
    #parser.add_argument("--save_interval", type=int, default=500)  # This is a number of EPISODES
    

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    run_options = get_args()
    trainDQN = TrainVanillaDqnV6(run_options)
    trainDQN.train()
