

class UnityEnvDecorator():
    def __init__(self, unity_environment):
        self.env = unity_environment
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        
        
    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        
        state = env_info.vector_observations
        
        return state
    
    
    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        
        return next_states, rewards, dones, env_info
        
        