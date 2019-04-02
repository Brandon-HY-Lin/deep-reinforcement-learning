
class Game()
    def __init__(self):
        self.agent_list = []
        self.index = 0
        
        
    def add_agent(self, agent):
        self.agent_list.append(agent)
        
        
    def __iter__(self)
        self.index = 0
        return self
    
    
    def __next__(self):
        if self.index < self.__len__():
            agent = agent_list[self.index]
            self.index += 1
            
            return agent
        else:
            raise StopIteration
            
            
    def __len__(self):
        return len(self.agent_list)