

class UnityEnvDecorator():
    def __init__(self, unity_environment):
        self.env = unity_environment
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        
        self.action_size = self.env.brains[self.brain_name].vector_action_space_size
        self.state_size = self.env.brains[self.brain_name].vector_observation_space_size
        
    

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        
        return next_state, reward, done, env_info


    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        state = env_info.vector_observations[0]
        
        return state
    
    
    def close(self):
        self.env.close()

        