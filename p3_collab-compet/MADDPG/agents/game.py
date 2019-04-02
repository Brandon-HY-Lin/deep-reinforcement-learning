
class Game():
    def __init__(self):
        self.agent_list = []
        self.index = 0
        
        
    def add_agent(self, agent):
        self.agent_list.append(agent)
        
        
    def index_of_agent(self, agent):
        for index, a in enumerate(self.agent_list):
            if a == agent:
                return index
            
        raise Expception('Cannot find agent which name is {}'.format(agent.name))
        
        
    def __iter__(self):
        self.index = 0
        return self
    
    
    def __next__(self):
        if self.index < self.__len__():
            agent = self.agent_list[self.index]
            self.index += 1
            
            return agent
        else:
            raise StopIteration
            
            
    def __len__(self):
        return len(self.agent_list)