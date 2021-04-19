import collections
import random

class ReplayMemory():
    
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
            
            oldest_game_id = -1 if len(self.game_ids) == 0 else self.game_ids[0]
            oldest_game_frame_count = 0 if oldest_game_id == -1 else self.game_frame_counts[oldest_game_id] 
            
            while new_buffer_size - oldest_game_frame_count >= self.frame_limit:
                del self.buffer[0:oldest_game_frame_count]
                del self.game_ids[0]
                del self.game_frame_counts[oldest_game_id]
                
                new_buffer_size = frame_count + len(self.buffer)
                oldest_game_id = -1 if len(self.game_ids) == 0 else self.game_ids[0]
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
