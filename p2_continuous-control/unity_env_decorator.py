

class UnityEnvDecorator():
    def __init__(self, unity_environment):
        self.env = unity_environment
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        
        self.action_size = self.env.brains[self.brain_name].vector_action_space_size
        self.state_size = self.env.brains[self.brain_name].vector_observation_space_size
        
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        # get number of agents
        self.num_agents = len(env_info.agents)
    

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        
        return next_states, rewards, dones, env_info


    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        states = env_info.vector_observations
        
        return states
    
    
    def close(self):
        self.env.close()

        