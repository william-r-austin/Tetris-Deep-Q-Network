'''
Created on Apr 11, 2021

@author: William
'''
from cs747.sim.tetris import Tetris
from cs747.sim.tetris_move_result import TetrisMoveResult
import random
from cs747.util.experience_replay import ReplayMemory

GAMMA = 0.9

if __name__ == '__main__':
    tetrisSim = Tetris(render_flag=False)
    action_names = tetrisSim.get_action_names()
    memory = ReplayMemory(frame_limit=8000)
    game_number = 1
    
    while game_number <= 1000:
        print("\nStarting new Tetris Game #" + str(game_number))
        
        action_count = 0
        game_over = False
        game_move_results = []
        current_state = tetrisSim.get_current_board_state()
        
        while not game_over:
            next_action = random.choice(action_names)
            result_map = tetrisSim.do_action_by_name(next_action)
            
            next_state = tetrisSim.get_current_board_state()
            game_over = result_map["gameover"]
            
            # state, action, reward, discounted_future_reward, next_state
            current_move_result = TetrisMoveResult(current_state, next_action, result_map["reward"], next_state)
            game_move_results.append(current_move_result)
            
            action_count += 1
            current_state = next_state
        
        print("Finished Game #" + str(game_number) + ". Total actions = " + str(action_count))
        
        discounted_reward = 0
        
        # Add a loop here to set the future discounted state for all actions
        # that we made during the game
        #
        # Note that the Q(s, a) value is: CurrentReward + DiscountedFutureRewards
        # So, we should add these values together when computing target values for the loss
        for move_result in reversed(game_move_results):
            move_result.discounted_future_reward = discounted_reward
            discounted_reward = GAMMA * (move_result.reward + discounted_reward)
            
        memory.insert(game_move_results, game_number)
        memory.print_state()
        
        game_number += 1
        tetrisSim.reset()
        
    print("Done playing tetris!!")