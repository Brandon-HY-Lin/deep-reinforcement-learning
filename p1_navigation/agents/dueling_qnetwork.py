import torch
import torch.nn as nn
import torch.nn.functional as F

class Dueling_QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super().__init__()
        
        self.seed = torch.manual_seed(seed)
        
        # advantage function (A(state, action))
        self.fc1_a = nn.Linear(state_size, fc1_units)
        self.fc2_a = nn.Linear(fc1_units, fc2_units)
        self.fc3_a = nn.Linear(fc2_units, action_size)
        
        # state function (V(state))
        self.fc1_v = nn.Linear(state_size, fc1_units)
        self.fc2_v = nn.Linear(fc1_units, fc2_units)
        self.fc3_v = nn.Linear(fc2_units, 1)
        
        
    def forward(self, state):
        x_a = F.relu(self.fc1_a(state))
        x_a = F.relu(self.fc2_a(x_a))
        x_a = self.fc3_a(x_a)
        
        x_v = F.relu(self.fc1_v(state))
        x_v = F.relu(self.fc2_v(x_v))
        x_v = self.fc3_v(x_v)

        x = x_v + x_a
        
        return x
        
