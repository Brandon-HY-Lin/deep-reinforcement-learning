import torch
import numpy as np

import pdb

from agents.base_agent import BaseAgent
from utils.experience_pack import pack_experience
from utils.replay_buffer import ReplayBuffer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AgentGroup(BaseAgent):
    def __init__(self, agent_list, action_size, learn_period=10, learn_sampling_num=20, buffer_size=int(1e6), batch_size=128, random_seed=0):
        super().__init__()
        
        if len(agent_list) == 0:
            raise Exception('len(agent_list) = 0')

        self.agent_list = agent_list
        
        self.learn_period = learn_period
        self.learn_sampling_num = learn_sampling_num
        
        self.batch_size = batch_size
        
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed, device)
        
        self.time_step = 0
        
        # debugging constant
        self.__debug_num_agents = len(agent_list)
        self.__debug_state_size = agent_list[0].state_size
        self.__debug_action_size = agent_list[0].action_size
        
 
    def act(self, states, add_noise=True):
        """
        Predict actions given states.
        
        Args:
            states (numpy.array): states.shape[0] = num_agents
            
        Returns:
            actions (numpy.array): actions.shape[0] = num_agents.
        """
#         assert (states.shape[0] == self.__debug_num_agents), 'Mismatch dim of states.shape[0]'
        
        actions = None
        
        for s, agent in zip(states, self.agent_list):
            
            s = np.expand_dims(s, axis=0)
#             pdb.set_trace()
            
            action = agent.act(s)
            
            # expand dim from (2,) to (1, 2)
            action = np.expand_dims(action, axis=0)

            if actions is None:
                actions = action
            else:
                actions = np.append(actions, action, axis=0)
              

#         pdb.set_trace()
        
#         assert (actions.shape[0] == self.__debug_num_agents), 'Mismatch dim of actions.shape[0]'
#         assert (actions.shape[0] == self.__debug_action_size), 'Mismatch dim of actions.shape[0]'
        
        return actions
    
    
    def step(self, states, actions, rewards, next_states, dones):
        
        # flatten states, action, rewards, next_states, dones
        p = pack_experience(states, actions, rewards, next_states, dones)
        
#         pdb.set_trace()
        
        self.memory.add(*p)
        
        if (len(self.memory) > self.batch_size) and (self.time_step % self.learn_period == 0):
            for _ in range(self.learn_sampling_num):
                for agent in self.agent_list:

#                     pdb.set_trace()
                    
                    # Note: experiences.shape[0] = batch_size
                    experiences = self.memory.sample()

                    agent.step(*experiences)
                    
                self.time_step += 1
                
                
    def reset(self):
        for agent in self.agent_list:
            agent.reset()


    def model_dicts(self):
        merged_dicts = {}
        
        for agent in self.agent_list:
            merged_dicts = {**merged_dicts, **agent.model_dicts()}
            
        return merged_dicts