'''
Created on Apr 18, 2021

@author: William
'''

class TetrisMoveResult():
	
	def __init__(self, begin_state, begin_tensor, begin_q_value, action, reward, final_state_flag, next_state, next_tensor, next_q_value, weight):
		self.begin_state = begin_state
		self.begin_tensor = begin_tensor
		self.begin_q_value = begin_q_value
		self.action = action
		self.reward = reward
		self.final_state_flag = final_state_flag
		self.next_state = next_state
		self.next_tensor = next_tensor
		self.next_q_value = next_q_value
		self.weight = weight
		self.discounted_future_reward = 0
		self.global_weight_sum = 0
	
	
	def set_discounted_future_reward(self, discounted_future_reward):
		self.discounted_future_reward = discounted_future_reward

	'''
	def get_begin_state(self,):
		return self.begin_state
	
	def get_action(self):
		return self.action
	''' 
