import random
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from networks.ddpg_actor_version_1 import DDPGActorVersion1
from networks.ddpg_critic_version_1 import DDPGCriticVersion1
from agents.base_agent import BaseAgent
from utils.ounoise import OUNoise
from utils.replay_buffer import ReplayBuffer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

""" DDPG Agent (Version 51)
    1. Reset OUNoise after every calling learn()
"""
class DDPGAgentVersion1(BaseAgent):
    def __init__(self, state_size, action_size, random_seed,
                lr_actor=1e-2, lr_critic=1e-2, 
                 fc1_units=128, fc2_units=128,
                 buffer_size=int(1e6), batch_size=50, 
                 gamma=0.95, tau=1e-2,
                 learn_period=100, learn_sampling_num=50,
                adam_critic_weight_decay=0.0, name=None):



        """Initialize an Agent object.
        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            
        """
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

#         self.max_norm = max_norm
        self.learn_period = learn_period
        self.learn_sampling_num = learn_sampling_num
        

        
        self.actor_local = DDPGActorVersion1(state_size, action_size, random_seed, 
                                             fc1_units=fc1_units, fc2_units=fc2_units).to(device)
        
        self.actor_target = DDPGActorVersion1(state_size, action_size, random_seed, 
                                              fc1_units=fc1_units, fc2_units=fc2_units).to(device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Network (w/ Target Network)
        self.critic_local = DDPGCriticVersion1(state_size, action_size, random_seed, 
                                               fcs1_units=fc1_units, fc2_units=fc2_units).to(device)
        
        self.critic_target = DDPGCriticVersion1(state_size, action_size, random_seed, 
                                                fcs1_units=fc1_units, fc2_units=fc2_units).to(device)
        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=adam_critic_weight_decay)
        # Noise process for action
        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15 # (Timothy Lillicrap, 2016)
        self.exploration_sigma = 0.2 # (Timothy Lillicrap, 2016)
        self.noise = OUNoise(action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
          
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed, device)


        self.gamma = gamma
        
        # soft update parameter
        self.tau = tau
        
        self.batch_size = batch_size
        
        self.name = name
        
        self.time_step = 0
        
        
    def step(self, state, action, reward, next_state, done):
        self.time_step += 1
        
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        
        # Learn, if enough samples are available in memory
        if (len(self.memory) > self.batch_size) and (self.time_step % self.learn_period == 0):
            for _ in range(self.learn_sampling_num):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
        
        
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
            
        return np.clip(action, -1, 1)
    
    
    def reset(self):
        self.noise.reset()
            
    
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        
        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        
        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # train critic
        # loss fuction = Q_target(TD 1-step boostrapping) - Q_local(current)
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        Q_targets = rewards + (gamma * Q_targets_next * (1 -dones))
        
        Q_expected = self.critic_local(states, actions)
        
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # train actor (policy gradient)
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
      
        
        # update critic_target
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        
        # update actor_target
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        
        #------ update noise ---#
        self.noise.reset()
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Args:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

            
    def model_dicts(self):
        return {'agent_{}_actor'.format(self.name) : self.actor_target,
                'agent_{}_critic'.format(self.name) : self.critic_target}
