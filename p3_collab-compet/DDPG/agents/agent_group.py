from agents.base_agent import BaseAgent

class AgentGroup(BaseAgent):
    def __init__(self, agent_list):
        super().__init__()
        
        self.agent_list = agent_list
        
        
    def act(self, states):
        actions = np.array([])
        
        for agent, state in zip(agent_list, states):
            actions = np.append(actions, agent.act(state), axis=0)
        
        return actions
    
    
    def step(self, states, actions, rewards, next_states, dones):
        for agent, state, action, reward, next_state, done in 
                    zip(agent_list, states, actions, rewards, next_states, dones):
                
            agent.step(state, action, reward, next_state, done)
    
    
    def reset(self):
        for agent in agent_list:
            agent.reset()