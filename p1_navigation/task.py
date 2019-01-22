from unityagents import UnityEnvironment

class Task():
    def __init__(self, file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
        '''
        Params:
        ======
            file_name: file of UnitEnvironment
        '''
        self.file_name = file_name
        self.env = UnityEnvironment(file_name=file_name)
        self.brain_name = self.env.brain_names[0]
        
        
    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        
        return next_state, reward, done, env_info
        
        
    def reset(self, train_mode=True):
        env_info = env.reset(train_mode=train_mode)[self.brain_name]
        state = env_info.vector_observations[0]   # get the current state
        
        return state
        