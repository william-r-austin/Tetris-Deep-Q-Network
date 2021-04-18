'''
Created on Apr 18, 2021

@author: William
'''

class TetrisMoveResult():
	
	def __init__(self, begin_state, action, reward, next_state):
		self.begin_state = begin_state
		self.action = action
		self.reward = reward
		self.next_state = next_state
		self.discounted_future_reward = 0 
	
	
	def set_discounted_future_reward(self, discounted_future_reward):
		self.discounted_future_reward = discounted_future_reward

	'''
	def get_begin_state(self,):
		return self.begin_state
	
	def get_action(self):
		return self.action
	''' 
