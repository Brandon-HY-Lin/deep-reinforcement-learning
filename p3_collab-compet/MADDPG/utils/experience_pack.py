
class ExperiencePack():
    """
        Flatten states from dim=(num_agent, state_size) to 
                                (state_size_1, state_size_2, ..., state_size_num_agent).
    """
    def __init__(self):
        pass
    
    def pack(self, states, actions, rewards, next_states, dones):
        return (np.flatten(states),
               np.flatten(actions),
               np.flatten(rewards),
               np.flatten(next_states),
               np.flatten(dones))
        

class ExperienceUnpack():
    """
        Unpack the experience.
    """
    
    def __init__(self, experiences, state_size, action_size, num_agents):
        self.experiences = experiences
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
    
    
    def __getitem__(self, key):
        if key >= self.num_agents:
            raise IndexError('Index > num_agents. The num_agents is {}.'.format(num_agents))
            
        states, actions, rewards, next_states, dones = self.experiences
        
        
        state_i = states[:, key*self.state_size:(key+1)*self.state_size]
        action_i = actions[:, key*self.action_size:(key+1)*self.action_size]
        reward_i = rewards[:, key]
        next_state_i = next_states[:, key*self.state_size:(key+1)*self.state_size]
        done_i = dones[:, key]
        
        return (state_i, action_i, reward_i, next_state_i, done_i)
        
        