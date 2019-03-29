import random
from collections import namedtuple, deque
import torch
import numpy as np
from utils.sum_tree import SumTree

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.memory = SumTree(buffer_size)  # internal memory (sum tree)
        self.__memory_buffer = []

        
        self.seed = random.seed(seed)
        self.device = device
        
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        #elf.memory.append(e)
        self.__memory_buffer.append(e)
        
        
    def add_batch(self, states, actions, rewards, next_states, dones):
        for s, a, r, next_s, d in zip(states, actions, rewards, next_states, dones):
            self.add(s, a, r, next_s, d)
            

    def sample(self):
        buf_len = len(self.__memory_buffer)
        mem_len = self.batch_size - buf_len

        experiences = []
        indices = []
        probs = []
        if mem_len:
            #segment = self.memory.total() / mem_len
            for i in range(mem_len):
                #s = random.uniform(segment * i, segment * (i + 1))
                s = random.uniform(0, self.memory.total())
                idx, p, e = self.memory.get(s)
                experiences.append(e)
                indices.append(idx)
                probs.append(p/self.memory.total())
        for e in self.__memory_buffer:
            # Add experience to the buffer and record its index
            experiences.append(e)
            #if self.__mode['PER']:
            idx = self.memory.add(0.0, e)  # Default value for p is 0
            indices.append(idx)
            probs.append(1/len(self))

        self.__memory_buffer.clear()

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones, indices, probs)
        

    def update(self, indices, priority_values):
        for idx, p in zip(indices, priority_values):
            self.memory.update(idx, p)

    def __len__(self):
        return max(len(self.memory), len(self.__memory_buffer))