"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn

class DeepQNetworkSimple(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepQNetworkSimple, self).__init__()
        self.a0 = nn.Sequential(nn.Linear(input_size, 120), nn.ReLU(inplace=True))
        self.a1 = nn.Sequential(nn.Linear(120, 72), nn.ReLU(inplace=True))
        self.a2 = nn.Sequential(nn.Linear(72, 43), nn.ReLU(inplace=True))
        self.a3 = nn.Sequential(nn.Linear(43, 26), nn.ReLU(inplace=True))
        self.a4 = nn.Sequential(nn.Linear(26, 16), nn.ReLU(inplace=True))
        self.a5 = nn.Sequential(nn.Linear(16, 9), nn.ReLU(inplace=True))
        self.a6 = nn.Sequential(nn.Linear(9, output_size))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.a0(x)
        x = self.a1(x)
        x = self.a2(x)
        x = self.a3(x)
        x = self.a4(x)
        x = self.a5(x)
        x = self.a6(x)

        return x

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        #nn.Conv2d(in_channels, out_channels, kernel_size, stride=4, padding, dilation, groups, bias, padding_mode)
        self.conv1 = nn.Sequential(nn.Linear(200, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 6))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

class DeepQNetworkAtari(nn.Module):
    def __init__(self, num_frames, num_actions):
        super(DeepQNetworkAtari, self).__init__()
        self.num_frames = num_frames
        self.num_actions = num_actions
        
        # Layers
        self.conv1 = nn.Conv2d(
            in_channels=num_frames,
            out_channels=16,
            kernel_size=8,
            stride=4,
            padding=2
            )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
            )
        self.fc1 = nn.Linear(
            in_features=3200,
            out_features=256,
            )
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=num_actions,
            )
        
        # Activation Functions
        self.relu = nn.ReLU()
    
    def flatten(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return x
    
    def forward(self, x):
        
        # Forward pass
        x = self.relu(self.conv1(x))  # In: (80, 80, 4)  Out: (20, 20, 16)
        x = self.relu(self.conv2(x))  # In: (20, 20, 16) Out: (10, 10, 32)
        x = self.flatten(x)           # In: (10, 10, 32) Out: (3200,)
        x = self.relu(self.fc1(x))    # In: (3200,)      Out: (256,)
        x = self.fc2(x)               # In: (256,)       Out: (4,)
        
        return x

class DeepQNetworkAtariSmall(nn.Module):
    def __init__(self, num_actions):
        super(DeepQNetworkAtariSmall, self).__init__()
        self.num_actions = num_actions
        
        # 
        self.convolutionalLayer1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
            )
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.leakyRelu1 = nn.LeakyReLU()
        
        '''
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=(1, 2),
            stride=(1, 2),
            padding=0
            )
        '''
        self.convolutionalLayer2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 2),
            stride=(4, 2),
            padding=0
            )
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.leakyRelu2 = nn.LeakyReLU()
        
        self.convolutionalLayer3 = nn.Conv2d(
            in_channels=64,
            out_channels=4,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.batchNorm3 = nn.BatchNorm2d(4)
        self.leakyRelu3 = nn.LeakyReLU()

        # Layer 4
        self.fullyConnectedLayer4 = nn.Linear(
            in_features=100,
            out_features=24,
        )
        self.leakyRelu4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout()
        
        
        self.fullyConnectedLayer5 = nn.Linear(
            in_features=24,
            out_features=num_actions,
        )
    
    def flatten(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return x
    
    def forward(self, x):
        '''
            Input to Layer 1: torch.Size([512, 1, 20, 10])
            Input to Layer 2: torch.Size([512, 32, 20, 10])
            Input to Layer 3: torch.Size([512, 64, 5, 5])
            Input to flatten: torch.Size([512, 4, 5, 5])
            Input to Layer 4: torch.Size([512, 100])
            Input to Layer 5: torch.Size([512, 24])
            Output: torch.Size([512, 6])
        '''
        
        
        #print("Input to Layer 1: " + str(x.size()))
        x = self.convolutionalLayer1(x)
        x = self.batchNorm1(x)
        x = self.leakyRelu1(x)
        
        #print("Input to Layer 2: " + str(x.size()))
        x = self.convolutionalLayer2(x)
        x = self.batchNorm2(x)
        x = self.leakyRelu2(x)
        
        #print("Input to Layer 3: " + str(x.size()))
        x = self.convolutionalLayer3(x)
        x = self.batchNorm3(x)
        x = self.leakyRelu3(x)
        
        #print("Input to flatten: " + str(x.size()))
        x = self.flatten(x)
        
        #print("Input to Layer 4: " + str(x.size()))
        x = self.fullyConnectedLayer4(x)
        x = self.leakyRelu4(x)
        x = self.dropout4(x)
        
        #print("Input to Layer 5: " + str(x.size()))
        x = self.fullyConnectedLayer5(x)
        
        #print("Output: " + str(x.size()))
        
        return x
