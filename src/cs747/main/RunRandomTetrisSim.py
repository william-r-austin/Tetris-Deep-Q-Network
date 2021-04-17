'''
Created on Apr 11, 2021

@author: William
'''
from cs747.sim.tetris import Tetris
import random

if __name__ == '__main__':
    possible_actions = ["MOVE_LEFT", "MOVE_RIGHT", "MOVE_DOWN", "DROP", "ROTATE_CLOCKWISE", "ROTATE_COUNTERCLOCKWISE"]
    tetrisSim = Tetris(render_flag=True)
    game_number = 1
    
    while game_number <= 100:
        print("\nStarting new Tetris Game #" + str(game_number))
        
        action_count = 0
        game_over = False
        
        while not game_over:
            next_action = random.choice(possible_actions)
            result_map = tetrisSim.doAction(next_action)
            action_count += 1
            #print("Action was:" + next_action + ". Result was " + str(result_map))
            game_over = result_map["gameover"]
        
        print("Finished Game #" + str(game_number) + ". Total actions = " + str(action_count))
        game_number += 1
        tetrisSim.reset()
        
    print("Done playing tetris!!")