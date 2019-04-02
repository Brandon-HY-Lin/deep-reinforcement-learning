import random
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from networks.maddpg_critic_version_1 import MADDPGCriticVersion1
from agents.base_agent import BaseAgent
from agents.game import Game
from utils import OUNoise

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ActionQuery()
    """
    Query result
    """
    def __init__(self):
        self.next_actions = None


""" 
MADDPGAgent (Version 1)
    1. Add gradient clipping of gradient of Q-function
    2. Reset OUNoise after every calling learn()
"""
class MADDPGAgenVersion1(BaseAgent):
    def __init__(self, game, state_size, action_size, random_seed=0,
                    lr_critic=1e-3, lr_actor=1e-3,
                    fc1_units=400, fc2_units=300,
                    buffer_size=int(1e6), batch_size=128,
                    gamma=0.99, tau=1e-3,
                    max_norm=1.0, learn_period=10, learn_sampling_num=20,
                    epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99):
        """Initialize an Agent object.
        Args:
            game (class Game): meidator in chain-of-responsibility design pattern. (Broker chain)
            random_seed (int): random seed.
            
            max_norm (float): value of clip_grad_norm for critic optimizer
        """
        super().__init__()
        
        self.agent_list = agent_list
        self.num_agents = len(agent_list)
        
        if (self.num_agents == 0):
            raise Exception('len(agent_list) = 0')
            
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.max_norm = max_norm
        self.learn_period = learn_period
        self.learn_sampling_num = learn_sampling_num
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epssilon_decay = epsilon_decay
        
        # Actor Network (w/ Target Network)
        self.actor_local = MADDPGActorVersion1(state_size, actions_size, random_seed, 
                                               fc1_units=fc1_units, fc2_unit=fc2_units).to(device)
        self.actor_target = MADDPGActorVersion1(state_size, actions_size, random_seed, 
                                                fc1_units=fc1_units, fc2_unit=fc2_units).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Network (w/ Target Network)
        self.critic_local = MADDPGCriticVersion1(state_size, action_size, random_seed, 
                                                 fcs1_units=fc1_units, fc2_units=fc2_units).to(device)
        self.critic_target = MADDPGCriticVersion1(state_size, action_size, random_seed, 
                                                  fcs1_units=fc1_units, fc2_units=fc2_units).to(device)
        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        
        # Noise process for action
        # Noise process
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

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

       
    def act(self, states, add_noise=True):
        """
            Returns actions for given state.
            The input size of actor networks is state_size.
        """
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
              
        return actions
    
    
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
            n_state = next_states[i*self.state_size: (i+1)*self.state_size]
            
            # predict next_action and append it to actionQuery.actions
            agent.query(n_state, q)
            
        return q.actions
    
    
    def query(self, next_state, q):
        """
        Args:
            q (class ActionQuery): parcel that stores actions
        """
        
        next_action = self.actor_local(next_state)
        
        if q.next_actions is None:
            q.next_actions = next_action
        else:
            q.next_actions = torch.cat((q.next_actions, next_action), dim=0)    
            q.next_actions = q.next_actions.flatten()


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
        
        for i, agent in enumerate(agent_list):
            # divide fields update agent number i
            experience_unpacks = ExperienceUnpack(e, self.state_size, self.action_size, self.num_agents)
            # upack field in agent_i
            states_i, actions_i, rewards_i, next_states_i, dones_i = experience_unpacks[i]
            
            assert (states_i.shape[1] == (self.state_size)), 'Wrong shape of states_i'
            assert (actions_i.shape[1] == (self.action_size)), 'Wrong shape of actions_i'
            assert (rewards_i.shape[1] == (1)), 'Wrong shape of rewards_i'
            assert (dones_i.shape[1] == (1)), 'Wrong shape of dones_i'
            
            # train critic
            # loss fuction = Q_target(TD 1-step boostrapping) - Q_local(current)      
            next_actions = self.forward_all(next_states)
            
            assert (next_actions.shape[1] == (self.action_size * num_agents)), 'Wrong shape of next_actions'
            
            Q_targets_next = self.critic_target(next_states, next_actions)

            Q_target_i = reward_i + (self.gamma * Q_targets_next * (1-done_i))
            Q_expected = self.critic_local(states, actions)
            
            critic_loss = F.mse_loss(Q_expected, Q_target_i)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.max_norm)
            self.critic_optimizer.step()

            # train actor
            actions_pred = self.get_all_actors_actions(states)
            actor_loss = - self.critic_local(states, actions_pred).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            
        # update critic
        self.soft_update(self.critic_local, self.critic_target, self.tau)

        # update actors in agents
        self.soft_update_actors()
        
        #------ update noise ---#
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
        self.noie.reset()
            
            
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
      
       
    def soft_update_actors(self):
        for agent in agent_list:
            self.soft_update(agent.actor_local, agent.actor_target, self.tau)
                                                     
    
    def model_dicts(self):
        m_dicts = {'critic': self.critic_target}
        
        for agent in agent_list:
            key = 'actor_{}'.format(agent.name)
            m_dicts[key] = agent.actor_target
        
        return m_dicts                                         
        
