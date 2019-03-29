class BaseAgent():
    def __init__(self):
        pass
    
    def act(self, states, add_noise=True):
        raise NotImplementedError('Error: In BaseAgent(), act() is not implemented')
        
    
    """
        Args:
            
    """
    def step(self, states, actions, rewards, next_states, dones):
        raise NotImplementedError('Error: In BaseAgent(), step() is not implemented')
    
    
    def reset(self):
        raise NotImplementedError('Error: In BaseAgent(), reset() is not implemented')

    
    """
        model_dict()
        
        Return:
            return name-model pair dict  example: ({'critic': critic_model, 
                                                    'actor': actor_model})
    """
    def model_dicts(self):
        raise NotImplementedError('Error: In BaseAgent(), model_dicts() is not implemented')
        