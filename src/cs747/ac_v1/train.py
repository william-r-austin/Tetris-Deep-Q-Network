'''
Created on Apr 29, 2021

@author: William
'''
from cs747.ac_v1.tetris import Tetris
from cs747.ac_v1.ac_model import ActorCriticModelV1
import numpy as np
import torch  
import torch.optim as optim
#from torch.autograd import Variable
#import matplotlib.pyplot as plt
#import pandas as pd

# hyperparameters
hidden_size = 256
learning_rate = 3e-4

# Constants
GAMMA = 0.99
max_episodes = 3000

#num_episodes = 20

def get_tensor_for_state(board_state):
    '''
    Convert the current board_state into a tensor for PyTorch
    '''
    state_tensor = torch.tensor(board_state, dtype=torch.float32).clip(-1, 1).unsqueeze(dim=0).unsqueeze(dim=0)
    return state_tensor


def a2c(env):
    #num_inputs = 200
    num_outputs = 6
    
    actor_critic = ActorCriticModelV1(num_outputs)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []
        done = False
        
        env.reset()
        state = env.get_current_board_state()
        state_tensor = get_tensor_for_state(state) 
        #torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(dim=0)
        
        while not done:
            value, policy_dist = actor_critic.forward(state_tensor)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            result_map = env.do_action_by_id(action)
            
            reward = result_map["reward"]
            new_state = env.get_current_board_state()
            new_state_tensor = get_tensor_for_state(new_state)
            #new_state_tensor = torch.tensor(new_state, dtype=torch.float32).flatten().unsqueeze(dim=0)
            done = result_map["gameover"]
            
            #new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            
            state = new_state
            state_tensor = new_state_tensor
            
            if done:
                Qval, _ = actor_critic.forward(new_state_tensor)
                Qval = Qval.detach().numpy()[0,0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(env.action_count)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:                    
                    print("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), env.action_count, average_lengths[-1]))
                break
        
        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
  
        #update actor critic
        values = torch.tensor(values)
        Qvals = torch.tensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()



if __name__ == '__main__':
    env = Tetris()
    
    episode = 1
    actions = env.get_action_names()
    
    '''
    while episode <= num_episodes:
        env.reset()
        print("-------------------------")
        print("Now running episode " + str(episode))
        print("-------------------------")
        
        while not env.gameover:
            
            action_index = np.random.randint(len(actions))
            print("Chose action: " + actions[action_index])
            result_map = env.do_action_by_id(action_index)
            print(result_map)
        
        episode += 1
    '''
         
    a2c(env)
    