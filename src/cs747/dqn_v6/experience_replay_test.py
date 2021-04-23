'''
Created on Apr 17, 2021

@author: William
'''
from cs747.dqn_v6.experience_replay import WeightedReplayMemory
import random

if __name__ == "__main__":
    memory = WeightedReplayMemory(capacity=10)
    
    index = 0
    while index < 26:
        entry = str(chr(index + 97))
        random_weight = random.random() * 10 + 5
        memory.insert(entry, random_weight)
        
        if memory.is_full():
            my_sample = memory.get_random_weighted_sample(5)
            print("Index = " + str(index) + ", Sample = " + str(my_sample))
            
        index += 1
    
    '''
    for game_id in range(100):
        game_size = random.randrange(8) + 2
        game_frames = [game_id] * game_size
        memory.insert(game_frames, game_id)
        print("Inserted data for Game ID = " + str(game_id))
        memory.print_state()
    '''    
        