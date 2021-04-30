
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticModel(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCriticModel, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        #state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

class ActorCriticModelV1(nn.Module):
    def __init__(self, num_actions):
        super(ActorCriticModelV1, self).__init__()

        '''
        This is the VALUE network.
        '''
        self.convolutionalLayer1a = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.batchNorm1a = nn.BatchNorm2d(32)
        self.leakyRelu1a = nn.LeakyReLU()
        
        self.convolutionalLayer2a = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(4, 2),stride=(4, 2),padding=0)
        self.batchNorm2a = nn.BatchNorm2d(64)
        self.leakyRelu2a = nn.LeakyReLU()
        
        self.convolutionalLayer3a = nn.Conv2d(in_channels=64,out_channels=4,kernel_size=1,stride=1,padding=0)
        self.batchNorm3a = nn.BatchNorm2d(4)
        self.leakyRelu3a = nn.LeakyReLU()
        
        # Layer 4
        self.fullyConnectedLayer4a = nn.Linear(in_features=100,out_features=24,)
        self.leakyRelu4a = nn.LeakyReLU()
        #self.dropout4a = nn.Dropout()

        self.fullyConnectedLayer5a = nn.Linear(in_features=24,out_features=1)
        
        
        '''
        This is the POLICY network.
        '''
        self.convolutionalLayer1b = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.batchNorm1b = nn.BatchNorm2d(32)
        self.leakyRelu1b = nn.LeakyReLU()
        
        self.convolutionalLayer2b = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(4, 2),stride=(4, 2),padding=0)
        self.batchNorm2b = nn.BatchNorm2d(64)
        self.leakyRelu2b = nn.LeakyReLU()
        
        self.convolutionalLayer3b = nn.Conv2d(in_channels=64,out_channels=4,kernel_size=1,stride=1,padding=0)
        self.batchNorm3b = nn.BatchNorm2d(4)
        self.leakyRelu3b = nn.LeakyReLU()
        
        # Layer 4
        self.fullyConnectedLayer4b = nn.Linear(in_features=100,out_features=24,)
        self.leakyRelu4b = nn.LeakyReLU()
        #self.dropout4b = nn.Dropout()
        
        self.fullyConnectedLayer5b = nn.Linear(in_features=24,out_features=num_actions)
        self.softmaxLayer5b = nn.Softmax(dim=1)
    
    def flatten(self, x):
        batch_size = x.size()[0]
        x = x.view(x.size()[0], -1)
        return x
    
    def forward(self, state):
        # Compute the state value
        value = self.convolutionalLayer1a(state)
        value = self.batchNorm1a(value)
        value = self.leakyRelu1a(value)
        
        value = self.convolutionalLayer2a(value)
        value = self.batchNorm2a(value)
        value = self.leakyRelu2a(value)
        
        value = self.convolutionalLayer3a(value)
        value = self.batchNorm3a(value)
        value = self.leakyRelu3a(value)
        
        #value = self.flatten(value)
        value = value.view(value.size()[0], -1)
        
        value = self.fullyConnectedLayer4a(value)
        value = self.leakyRelu4a(value)
        #value = self.dropout4a(value)
        
        value = self.fullyConnectedLayer5a(value)
        
        # Compute the policy distribution for the state
        policy_dist = self.convolutionalLayer1b(state)
        policy_dist = self.batchNorm1b(policy_dist)
        policy_dist = self.leakyRelu1b(policy_dist)
        
        policy_dist = self.convolutionalLayer2b(policy_dist)
        policy_dist = self.batchNorm2b(policy_dist)
        policy_dist = self.leakyRelu2b(policy_dist)
        
        policy_dist = self.convolutionalLayer3b(policy_dist)
        policy_dist = self.batchNorm3b(policy_dist)
        policy_dist = self.leakyRelu3b(policy_dist)
        
        policy_dist = policy_dist.view(policy_dist.size()[0], -1)
        #policy_dist = self.flatten(policy_dist)
        
        policy_dist = self.fullyConnectedLayer4b(policy_dist)
        policy_dist = self.leakyRelu4b(policy_dist)
        #policy_dist = self.dropout4b(policy_dist)
        
        policy_dist = self.fullyConnectedLayer5b(policy_dist)
        policy_dist = self.softmaxLayer5b(policy_dist)

        return value, policy_dist