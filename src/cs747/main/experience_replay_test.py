'''
Created on Apr 17, 2021

@author: William
'''
from cs747.util.experience_replay import ReplayMemory
import random

if __name__ == "__main__":
    memory = ReplayMemory(frame_limit=50)
    memory.print_state()
    
    for game_id in range(100):
        game_size = random.randrange(8) + 2
        game_frames = [game_id] * game_size
        memory.insert(game_frames, game_id)
        print("Inserted data for Game ID = " + str(game_id))
        memory.print_state()
        
        