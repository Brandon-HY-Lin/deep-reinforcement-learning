import random
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from networks.ddpg_actor_version_2 import DDPGActorVersion2
from networks.ddpg_critic_version_2 import DDPGCriticVersion2
from agents.base_agent import BaseAgent
# from agents.ounoise import OUNoise
from agents.ounoise_multivariate import OUNoiseMultivariate
from agents.replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""" DDPG Agent (Version 5)
    1. Add gradient clipping of gradient of Q-function
    2. Change learn period from 1 time step to 20 time steps.
        In each period, change the number of sampling from 1 to 10.
    3. Add exploration decay parameters:
        epsilon=1.0
        epsilon_decay = 1e-6
        After calling learn(), epsilon -= epsilon_decay 
    4. Reset OUNoise after every calling learn()
"""
class DDPGAgentVersion5(BaseAgent):
    def __init__(self, state_size, action_size, num_agents, random_seed,
                 lr_actor=1e-4, lr_critic=1e-3, 
                 fc1_units=400, fc2_units=300,
                 buffer_size=int(1e5), batch_size=128,
                 gamma=0.99, tau=1e-3,
                max_norm=1.0, learn_period=20, learn_sampling_num=10,
                epsilon=1.0, epsilon_decay=1e-6,
                 adam_critic_weight_decay=0.0):
                 
        """Initialize an Agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            
            max_norm (float): value of clip_grad_norm for critic optimizer
        """

        super().__init__()
        
        self.state_size = state_size
        self.num_agents = num_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.max_norm = max_norm
        self.learn_period = learn_period
        self.learn_sampling_num = learn_sampling_num
        self.epsilon = epsilon
        self.epsilon_decay = 1e-6
        
        # Actor Network (w/ Target Network)
        self.actor_local = DDPGActorVersion2(state_size, action_size, random_seed, fc1_units=fc1_units, fc2_units=fc2_units).to(device)
        self.actor_target = DDPGActorVersion2(state_size, action_size, random_seed, fc1_units=fc1_units, fc2_units=fc2_units).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Network (w/ Target Network)
        self.critic_local = DDPGCriticVersion2(state_size, action_size, random_seed, fcs1_units=fc1_units, fc2_units=fc2_units).to(device)
        self.critic_target = DDPGCriticVersion2(state_size, action_size, random_seed, fcs1_units=fc1_units, fc2_units=fc2_units).to(device)
        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=adam_critic_weight_decay)
        
        # Noise process for action
        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15 # (Timothy Lillicrap, 2016)
        self.exploration_sigma = 0.2 # (Timothy Lillicrap, 2016)
#         self.noise = OUNoise(action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        self.noise = OUNoiseMultivariate((num_agents, action_size), random_seed, 
                                         mu=self.exploration_mu, 
                                         theta=self.exploration_theta, 
                                         sigma=self.exploration_sigma)
    
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed, device)
        
        # parameter of discounted reward
        self.gamma = gamma
        
        # soft update parameter
        self.tau = tau
        
        self.batch_size = batch_size     
        
        
    def step(self, states, actions, rewards, next_states, dones, time_step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(self.num_agents):
            self.memory.add(states[i,:], actions[i,:], rewards[i], next_states[i,:], dones[i])
        #self.memory.add_batch(states, actions, rewards, next_states, dones)
        
        # Learn, if enough samples are available in memory
        if (len(self.memory) > self.batch_size) and (time_step%self.learn_period == 0):
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
            action += self.epsilon * self.noise.sample()
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
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.max_norm)
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
        self.epsilon -= self.epsilon_decay
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
        return {'actor': self.actor_target,
                'critic': self.critic_target}
