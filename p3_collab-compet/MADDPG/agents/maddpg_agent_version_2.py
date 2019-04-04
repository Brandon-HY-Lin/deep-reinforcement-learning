import random
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from networks.maddpg_critic_version_1 import MADDPGCriticVersion1
from networks.maddpg_actor_version_1 import MADDPGActorVersion1
from agents.base_agent import BaseAgent
from agents.game import Game
from utils.ounoise import OUNoise
from utils.experience_pack import ExperienceUnpack

import pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ActionQuery():
    """
    Query result
    """
    def __init__(self):
        self.next_actions = None


""" 
MADDPGAgent (Version 2)
    1. Add gradient clipping of gradient of Q-function
    2. Reset OUNoise after every calling learn()
    3. In forward_all, if agent_other is not self, detach tensor of agent_other.forward()
"""
class MADDPGAgentVersion2(BaseAgent):
    def __init__(self, game, num_agents, state_size, action_size, name, random_seed=0,
                    lr_critic=1e-3, lr_actor=1e-3,
                    fc1_units=400, fc2_units=300,
                    buffer_size=int(1e6), batch_size=128,
                    gamma=0.99, tau=1e-3,
                    max_norm=1.0,
                    epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99,
                    exploration_mu=0.0, exploration_theta=0.15, exploration_sigma=0.2):
        
        """Initialize an Agent object.
        Args:
            game (class Game): meidator in chain-of-responsibility design pattern. (Broker chain)
            random_seed (int): random seed.
            
            max_norm (float): value of clip_grad_norm for critic optimizer
        """
        super().__init__()
        
        self.index_agent = None
        
        self.game = game
        self.num_agents = num_agents
            
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.seed = random.seed(random_seed)
        
        self.max_norm = max_norm
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Actor Network (w/ Target Network)
        self.actor_local = MADDPGActorVersion1(state_size, action_size, random_seed, 
                                               fc1_units=fc1_units, fc2_units=fc2_units).to(device)
        self.actor_target = MADDPGActorVersion1(state_size, action_size, random_seed, 
                                                fc1_units=fc1_units, fc2_units=fc2_units).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Network (w/ Target Network)
        self.critic_local = MADDPGCriticVersion1(num_agents, state_size, action_size, 
                                                 fcs1_units=fc1_units, fc2_units=fc2_units,
                                                 seed=random_seed).to(device)
        self.critic_target = MADDPGCriticVersion1(num_agents, state_size, action_size, 
                                                  fcs1_units=fc1_units, fc2_units=fc2_units,
                                                 seed=random_seed).to(device)
        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        
        # Noise process for action
        # Noise process
        self.noise = OUNoise(self.action_size, exploration_mu, exploration_theta, exploration_sigma)

        # parameter of discounted reward
        self.gamma = gamma
        
        # soft update parameter
        self.tau = tau
        
        self.batch_size = batch_size

        
    def step(self, states, actions, rewards, next_states, dones):
        """
        Args:
            states (numpy.array): states.shape[1] = (state_size*num_agents)
            actions (numpy.array): actions.shape[1] = (actions_size*num_agents)
            next_states (numpy.array): next_states.shape[1] = (state_size*num_agents)
        """
        
        self.learn(states, actions, rewards, next_states, dones)

       
    def act(self, state, add_noise=True):
        """
            Returns actions for given state.
            The input size of actor networks is state_size.
        """
        
        state = torch.from_numpy(state).float().to(device)
        
        with torch.no_grad(): 
            self.actor_local.eval()

            action = self.actor_local(state).cpu().data.numpy()

            self.actor_local.train()

            if add_noise:
                action += self.epsilon * self.noise.sample()

        return np.clip(action, -1, 1)    
    
    
    def reset(self):
        self.noise.reset()

        
    def forward_all(self, next_states):
        """
        Get next_actions. This is a chain-of-responsibility design pattern. (Broker chain)
        
        Return:
            1d differentiable tensor of next_actions.
        """
        q = ActionQuery()
        
        for i, agent in enumerate(self.game):
            # get next_state_i of agent_i
            n_state = next_states[:, i*self.state_size: (i+1)*self.state_size]
            
#             pdb.set_trace()
            
            if agent == self:
                detach = False
            else:
                detach = True
                
            # predict next_action and append it to actionQuery.actions
            agent.query(n_state, q, detach)
            
        return q.next_actions
    
    
    def query(self, next_state, q, detach):
        """
        Args:
            q (class ActionQuery): parcel that stores actions
        """
        
        next_action = self.actor_local(next_state)
        
        if detach is True:
            next_action = next_action.detach()
        
        if q.next_actions is None:
            q.next_actions = next_action
        else:
            q.next_actions = torch.cat((q.next_actions, next_action), dim=1)    
            
#             pdb.set_trace()


    def learn(self, states, actions, rewards, next_states, dones):
        """Update policy and value parameters using given batch of experience tuples.
        For agent i:
            Q_target_i = r_i + gamma * critic_target(next_state, actor_target(next_state))
            
        where:
            actor_target(state) -> actions for all agent
            critic_target(state, action) -> Q-value
        
        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        # divide fields update agent number i
        experience_unpacks = ExperienceUnpack(states, actions, rewards, next_states, dones,
                                              self.state_size, self.action_size, self.num_agents)
        
        # upack field in agent_i
        if self.index_agent is None:
            self.index_agent = self.game.index_of_agent(self)
            
            
#         pdb.set_trace()
            
        states_i, actions_i, rewards_i, next_states_i, dones_i = experience_unpacks[self.index_agent]

#         assert (states_i.shape[1] == (self.state_size)), 'Wrong shape of states_i'
#         assert (actions_i.shape[1] == (self.action_size)), 'Wrong shape of actions_i'
#         assert (rewards_i.shape[1] == (1)), 'Wrong shape of rewards_i'
#         assert (dones_i.shape[1] == (1)), 'Wrong shape of dones_i'

        # train critic
        # loss fuction = Q_target(TD 1-step boostrapping) - Q_local(current)      
        next_actions = self.forward_all(next_states)

        assert (next_actions.shape[1] == (self.action_size * self.num_agents)), 'Wrong shape of next_actions'

        Q_targets_next = self.critic_target(next_states, next_actions)

        Q_target_i = rewards_i + (self.gamma * Q_targets_next * (1-dones_i))
        Q_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_target_i)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.max_norm)
        self.critic_optimizer.step()

        # train actor
        actions_pred = self.forward_all(states)
        actor_loss = - self.critic_local(states, actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

            
        # update critic
        self.soft_update(self.critic_local, self.critic_target, self.tau)

        # update actors
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        
        #------ update noise ---#
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
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
        m_dicts = {'critic_{}'.format(self.name): self.critic_target,
                   'actor_{}'.format(self.name): self.actor_target}
        
        return m_dicts                                         
        
