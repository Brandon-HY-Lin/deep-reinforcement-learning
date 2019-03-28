import numpy as np

from agents.base_agent import BaseAgent

import pdb

class AgentGroup(BaseAgent):
    def __init__(self, agent_list):
        super().__init__()
        
        self.agent_list = agent_list
        
        
    def act(self, states):
        combined_actions = None
        
        for agent, state in zip(self.agent_list, states):
            action = np.expand_dims(agent.act(state), axis=0)
            
            if combined_actions is None:
                combined_actions = action
            else:
                combined_actions = np.append(combined_actions, action, axis=0)
            
#             pdb.set_trace()
        
        return combined_actions
    
    
    def step(self, states, actions, rewards, next_states, dones):
        for agent, state, action, reward, next_state, done in zip(self.agent_list, 
                                                                  states, actions, rewards, next_states, dones):
                
#             pdb.set_trace()
            agent.step(state, action, reward, next_state, done)
    
    
    def reset(self):
        for agent in self.agent_list:
            agent.reset()
            
            
    def model_dicts(self):
        merged_dicts = {}
        
        for agent in self.agent_list:
            merged_dicts = {**merged_dicts, **agent.model_dicts()}
            
        return merged_dicts