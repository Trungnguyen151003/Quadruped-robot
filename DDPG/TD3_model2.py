import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from DDPG.network import ActorNetwork, CriticNetwork
from DDPG.utils import Memory

class TD3agent:
    def __init__(self, env, hidden_size1=256, hidden_size2=128,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, max_memory_size=100000):
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        # Actor
        self.actor = ActorNetwork(self.num_states, hidden_size1, hidden_size2, self.num_actions)
        self.actor_target = ActorNetwork(self.num_states, hidden_size1, hidden_size2, self.num_actions)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic 1
        self.critic_1 = CriticNetwork(self.num_states + self.num_actions, hidden_size1, hidden_size2, 1)
        self.critic_target_1 = CriticNetwork(self.num_states + self.num_actions, hidden_size1, hidden_size2, 1)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=critic_lr)

        # Critic 2
        self.critic_2 = CriticNetwork(self.num_states + self.num_actions, hidden_size1, hidden_size2, 1)
        self.critic_target_2 = CriticNetwork(self.num_states + self.num_actions, hidden_size1, hidden_size2, 1)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        return action

    def update(self, batch_size):
        self.total_it += 1
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.float32(dones))

        # Select action according to policy and add clipped noise
        noise = torch.FloatTensor(actions).data.normal_(0, self.policy_noise)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_states) + noise).clamp(-1, 1)

        # Compute the target Q value
        target_Q1 = self.critic_target_1(next_states, next_action)
        target_Q2 = self.critic_target_2(next_states, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (1 - dones) * self.gamma * target_Q.detach()

        # Get current Q estimates
        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)

        # Compute critic loss
        critic_loss_1 = self.critic_criterion(current_Q1, target_Q)
        critic_loss_2 = self.critic_criterion(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
