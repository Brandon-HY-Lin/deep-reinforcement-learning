import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network_utils import hidden_init

class MADDPGCriticVersion4(nn.Module):
    def __init__(self, num_agents, state_size, action_size, fcs1_units, fc2_units):
        """Initialize parameters and build model.
        Params
        ======
            num_agents (int): number of agents
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super().__init__()
        
#         self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(num_agents*(state_size+action_size) , fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
                             
        self.reset_parameters()
         
    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        
    def forward(self, states_1d, actions_1d):
        """
         Build a critic (value) network that maps (state, action) pairs -> Q-values.
         
         Args:
            states_1d (torch.tensor): shape[1] = (num_agents*state_size)
            actions_1d (torch.tensor): shape[1] = (num_agents*action_size)
        """
        x = torch.cat((states_1d, actions_1d), dim=1)
        
        x = F.relu(self.fcs1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)
