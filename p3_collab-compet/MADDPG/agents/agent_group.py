
from agents.base_agent import BaseAgent
from utils.experience_pack import ExperiencePack

class AgentGroup(BaseAgent):
    def __init__(self, agent_list, buffer_size=int(1e6), batch_size=128):
        super().__init__()
        
        if len(agent_list) == 0:
            raise Exception('len(agent_list) = 0')

        self.agent_list = agent_list
        
        self.batch_size = batch_size
        
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
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
        assert (states.shape[0] == self.__debug_num_agents), 'Mismatch dim of states.shape[0]'
        
        actions = None
        
        with torch.no_grad():
            for s, agent in zip(states, agent_list):
                
                agent.actor_local.eval()
                
                action = agent.act(s).cpu().data.numpy()
                
                agent.actor_local.train()
                
                if add_noise:
                    action += self.epsilon * self.noise.sampel()
                    action = np.clip(action, -1, 1)    
                
                if actions is None:
                    actions = action
                else:
                    actions = np.append(actions, action, axis=0)
              

        assert (actions.shape[0] == self.__debug_num_agents), 'Mismatch dim of actions.shape[0]'
        
        return actions
    
    
    def step(self, states, actions, rewards, next_states, dones):
        
        # flatten states, action, rewards, next_states, dones
        self.memory.add(ExperiencePack(states, actions, rewards, next_states, dones).pack())
        
        if (len(self.memory) > self.batch_size) and (self.time_step % self.learn_period == 0):
            for _ in range(self.learn_sampling_num):
                for agent in agent_list:

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