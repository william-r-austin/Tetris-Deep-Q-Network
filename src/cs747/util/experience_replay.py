import collections
import random

class ReplayMemory():
    
    def __init__(self, frame_limit=50):
        self.frame_limit = frame_limit
        self.game_frame_counts = {}
        self.game_ids = []
        self.buffer = []
                
    def insert(self, game_frames, game_id):
        frame_count = len(game_frames)
        new_buffer_size = frame_count + len(self.buffer)
        
        while new_buffer_size > self.frame_limit:
            oldest_game_id = self.game_ids[0]
            oldest_game_frame_count = self.game_frame_counts[oldest_game_id]
            
            del self.buffer[0:oldest_game_frame_count]
            del self.game_ids[0]
            del self.game_frame_counts[oldest_game_id]
            
            new_buffer_size = frame_count + len(self.buffer)
        
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
