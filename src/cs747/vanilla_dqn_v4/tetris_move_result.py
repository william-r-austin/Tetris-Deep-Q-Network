'''
Created on Apr 18, 2021

@author: William
'''

class TetrisMoveResult():
	
	def __init__(self, begin_state, begin_tensor, action, reward, final_state_flag, next_state, next_tensor):
		self.begin_state = begin_state
		self.begin_tensor = begin_tensor
		self.action = action
		self.reward = reward
		self.final_state_flag = final_state_flag
		self.next_state = next_state
		self.next_tensor = next_tensor
		self.discounted_future_reward = 0 
	
	
	def set_discounted_future_reward(self, discounted_future_reward):
		self.discounted_future_reward = discounted_future_reward

	'''
	def get_begin_state(self,):
		return self.begin_state
	
	def get_action(self):
		return self.action
	''' 
