import numpy as np
import random
from collections import namedtuple, deque

from qnetwork import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


class DQNAgent():
    
    def __init__(self, state_size, action_size, seed,
                lr=5e-4, buffer_size=1e5, batch_size=64, update_every=4,
                gamma=0.99, tau=1e-3):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.lr = lr
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        
    def step(self, state, action, reward, next_sate, done):
        
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY teim steps.
        self.t_step = (self.t_step + 1) % self.update_every 
        
        if self.t_step == 0:
            
            # If ennough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
                
                
    def act(self, state, eps=0.):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            
        self.qnetwork_local.train()
        
        
        # epsilon-greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arnage(self.action_size))
        
        
    def learn(self, experiences, gamma):
        
        state, actions, reward, next_states, dones = experiences
        
        Q_targets_next = self.nqnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
                               
        Q_expected = self.qnetwork_local(states).gather(1, actions)
                               
        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer.zeros_grad()
        loss.backward()
        
        self.optimizer.step()
        
        ###### update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        
        
    def soft_update(self, local_model, target_model, tau):
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau) * target_param.data)