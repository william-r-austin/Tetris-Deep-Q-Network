'''
Created on Apr 18, 2021

@author: William
'''

import random
from collections import deque
import numpy as np

# This class inserts entire games
class ReplayMemory(object):
    
    def __init__(self, frame_limit=50):
        self.frame_limit = frame_limit
        self.game_frame_counts = {}
        self.game_ids = []
        self.buffer = []
    
    def is_full(self):
        return len(self.buffer) >= self.frame_limit
    
                
    def insert(self, game_frames, game_id):
        
        frame_count = len(game_frames) 
        if frame_count > 0:
            new_buffer_size = frame_count + len(self.buffer)
            game_count = len(self.game_ids)
            oldest_game_id = -1 if game_count == 0 else self.game_ids[0]
            oldest_game_frame_count = 0 if oldest_game_id == -1 else self.game_frame_counts[oldest_game_id] 
            
            while game_count > 0 and new_buffer_size - oldest_game_frame_count >= self.frame_limit:
                del self.buffer[0:oldest_game_frame_count]
                del self.game_ids[0]
                del self.game_frame_counts[oldest_game_id]
                
                new_buffer_size = frame_count + len(self.buffer)
                game_count = len(self.game_ids)
                oldest_game_id = -1 if game_count == 0 else self.game_ids[0]
                oldest_game_frame_count = 0 if oldest_game_id == -1 else self.game_frame_counts[oldest_game_id]
            
            self.game_ids.append(game_id)
            self.game_frame_counts[game_id] = frame_count
            self.buffer.extend(game_frames)
    
    def get_random_sample(self, sample_size):
        return random.sample(self.buffer, k=sample_size)
    
    def get_size(self):
        return len(self.buffer)
    
    def print_state(self):
        print("Replay Memory Info. Size = " + str(len(self.buffer)))
        print("Game IDs = " + str(self.game_ids))
        print("Game ID Counts = " + str(self.game_frame_counts))
        

'''
    This class supports weighted sampling
'''
class WeightedReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.object_buffer = deque(maxlen=self.capacity)
        self.weight_offsets = deque(maxlen=self.capacity)
        self.total_buffer_weight = 0
        self.global_weight_sum = 0
    
    def is_full(self):
        return len(self.object_buffer) == self.capacity
    
    def get_total_weight(self):
        return self.total_buffer_weight
    
    def get_size(self):
        return len(self.object_buffer)
    
    def get_capacity(self):
        return self.capacity
    
    def insert(self, new_object, weight):
        new_object_tuple = (new_object, weight, self.global_weight_sum)
        
        if(len(self.object_buffer) < self.capacity):
            self.object_buffer.append(new_object_tuple)
            self.total_buffer_weight += weight
            self.global_weight_sum += weight
        else:
            removed_object = self.object_buffer.popleft()
            removed_weight = removed_object[1]
            self.total_buffer_weight -= removed_weight
            
            self.object_buffer.append(new_object_tuple)
            self.total_buffer_weight += weight
            self.global_weight_sum += weight
    
    def get_random_weighted_sample(self, sample_size):
        if len(self.object_buffer) > 0:
            offset = self.object_buffer[0][2]
            random_weights = self.total_buffer_weight * np.random.random_sample((sample_size,)) + offset
            sample_indices = [self.find_index_for_weight(random_weights[i]) for i in range(sample_size)]
            return [self.object_buffer[k][0] for k in sample_indices]
        
        return None     
    
    def find_index_for_weight(self, sample_weight):
        start_index = 0
        end_index = len(self.object_buffer) - 1
        
        while end_index > start_index:
            difference = end_index - start_index
            mid_index = start_index + (difference // 2) + (1 if difference % 2 > 0 else 0)
            mid_object = self.object_buffer[mid_index]
            mid_val = mid_object[2] 
            
            if mid_val < sample_weight:
                start_index = mid_index
            elif mid_val >= sample_weight:
                end_index = mid_index - 1
        
        return start_index
    
