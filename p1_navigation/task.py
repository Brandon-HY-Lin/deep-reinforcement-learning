from unityagents import UnityEnvironment

class Task():
    def __init__(self, unity_env):
        '''
        Params:
        ======
            file_name: file of UnitEnvironment
        '''
        self.env = unity_env
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
        state = env_info.vector_observations[0]   # get the current state
        
        return state
    
    
    def close(self):
        self.env.close()
        
        
    def get_action_size(self):
        return self.brain.vector_action_space_size
    
    
    def get_state_size(self):
        return self.brain.vector_observation_space_size
        