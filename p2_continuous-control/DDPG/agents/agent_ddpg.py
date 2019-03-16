import random
from ..models.ddpg_actor import DDPGActor
from ..models.ddpg_critic import DDPGCritic
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class AgentDDPG(AgentSaving):
    def __init__(self, state_size, action_size, random_seed,
                 lr_actor=1e-4, lr_critic=1e-3, 
                 buffer_size=int(1e5), batch_size=128,
                 gamma=0.99, tau=1e-3):
                 
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        # Actor Network
        self.actor_local = DDPGActor(state_size, action_size, random_seed).to(device)
        self.actor_target = DDPGActor(state_size, action_size, random_seed).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Network
        self.critic_local = DDPGCritic(state_size, action_size, random_seed).to(device)
        self.critic_target = DDPGCritic(state_size, action_size, random_seed).to(device)
        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        
        # Noise process for action
        self.noise = OUNoise(action_size, random_seed)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)
        
        # parameter of discounted reward
        self.gamma = gamma
        
        # soft update parameter
        self.tau = tau
        
        self.batch_size = batch_size
            
        #self.hard_update(self.actor_local, self_actor_target)
        
        
        
    def step(self, state, action, reward, next_state, done):
        
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences,self.gamma)
       
        
    def act(self, state, add_noise=True):
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
        states, actions, rewards, next_states, dones = experiences
        
        # train critic
        # loss fuction = Q_target(TD 1-step boostrapping) - Q_local(current)
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 -dones))
        
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
        
        
    def soft_update(self, local_model, target_model, tau):
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data
