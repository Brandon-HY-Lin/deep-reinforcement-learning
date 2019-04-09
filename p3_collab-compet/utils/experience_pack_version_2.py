import numpy as np
import torch

import pdb

def pack_experience(states, actions, rewards, next_states, dones):
    """
        Flatten states from dim=(num_agent, state_size) to 
                                (state_size_1, state_size_2, ..., state_size_num_agent).
    """

#     pdb.set_trace()
        
    return (states.flatten(),
           actions.flatten(),
           rewards,
           next_states.flatten(),
           dones)
        

class ExperienceUnpackVersion2():
    """
        Unpack the experience.
    """
    
    def __init__(self, states, actions, rewards, next_states, dones, 
                         state_size, action_size, num_agents):
        
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
    
    
    def __getitem__(self, key):
        if key >= self.num_agents:
            raise IndexError('Index > num_agents. The num_agents is {}.'.format(num_agents))
            
#         pdb.set_trace()
        
        state_i = self.states[:, key*self.state_size:(key+1)*self.state_size]
        action_i = self.actions[:, key*self.action_size:(key+1)*self.action_size]
        reward_i = self.rewards[:, key]
        next_state_i = self.next_states[:, key*self.state_size:(key+1)*self.state_size]
        done_i = self.dones[:, key]
        
        reward_i = torch.unsqueeze(reward_i, dim=1)
        done_i = torch.unsqueeze(done_i, dim=1)
        
        return (state_i, action_i, reward_i, next_state_i, done_i)
        
        