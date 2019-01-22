import torch

class ReplayBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        
        '''
        Params
        ======
            action_size (int): dimension of each action
        '''
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
    def add(self, state, action, reward, next_state, done):
        
        e = self.experience(state, aciton, reward, next_state, done)
        self.memory.append(e)
        
        
    def sampel(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack[e.done for e in experiences if e is not None])).astype(np.unit8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
    
    def __len__(self):
        return len(self.memory)