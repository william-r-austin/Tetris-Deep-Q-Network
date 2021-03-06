"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn

class DeepQNetworkSimple(nn.Module):
    def __init__(self):
        super(DeepQNetworkSimple, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(200, 64), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(32, 16), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Linear(16, 6))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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