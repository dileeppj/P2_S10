# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import time
import matplotlib.pyplot as plt
from collections import deque

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.conv1 = nn.Conv2d(1, 8, 3, 1,padding=1)
    self.conv2 = nn.Conv2d(8, 16, 3, 1,padding=1)
    self.conv3 = nn.Conv2d(16, 16, 3, 1,padding=1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.avgpooling = nn.AvgPool2d(20)
    self.fc1 = nn.Linear(18, 32)
    self.fc2 = nn.Linear(32, action_dim)
    self.max_action = max_action

  def forward(self, state1, state2):
    x = self.conv1(state1)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = self.conv3(x)
    x = self.avgpooling(x)
    x = torch.flatten(x, 1)
    x = torch.cat([x, state2], 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    actions = self.fc2(x)
    return self.max_action * torch.tanh(actions)

class Critic(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    self.conv1 = nn.Conv2d(1, 8, 3, 1,padding=1)
    self.conv2 = nn.Conv2d(8, 16, 3, 1,padding=1)
    self.conv3 = nn.Conv2d(16, 16, 3, 1,padding=1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.avgpooling1 = nn.AvgPool2d(20)
    self.fc1 = nn.Linear(19, 32)
    self.fc2 = nn.Linear(32, 1)
    
    self.conv4 = nn.Conv2d(1, 8, 3, 1,padding=1)
    self.conv5 = nn.Conv2d(8, 16, 3, 1,padding=1)
    self.conv6 = nn.Conv2d(16, 16, 3, 1,padding=1)
    self.dropout3 = nn.Dropout2d(0.25)
    self.dropout4 = nn.Dropout2d(0.5)
    self.avgpooling2 = nn.AvgPool2d(20)
    self.fc3 = nn.Linear(19, 32)
    self.fc4 = nn.Linear(32, 1)

  def forward(self, state1, state2, u):
    x1 = self.conv1(state1)
    x1 = F.relu(x1)
    x1 = self.conv2(x1)
    x1 = F.relu(x1)
    x1 = F.max_pool2d(x1, 2)
    x1 = self.dropout1(x1)
    x1 = self.conv3(x1)
    x1 = self.avgpooling1(x1)
    x1 = torch.flatten(x1, 1)
    x1 = torch.cat([x1,state2, u], 1)
    x1 = self.fc1(x1)
    x1 = F.relu(x1)
    x1 = self.dropout2(x1)
    x1 = self.fc2(x1)

    x2 = self.conv4(state1)
    x2 = F.relu(x2)
    x2 = self.conv5(x2)
    x2 = F.relu(x2)
    x2 = F.max_pool2d(x2, 2)
    x2 = self.dropout3(x2)
    x2 = self.conv6(x2)
    x2 = self.avgpooling2(x2)
    x2 = torch.flatten(x2, 1)
    x2 = torch.cat([x2, state2, u], 1)
    x2 = self.fc3(x2)
    x2 = F.relu(x2)
    x2 = self.dropout4(x2)
    x2 = self.fc4(x2)
    return x1, x2

  def Q1(self, state1, state2, u):
    x1 = self.conv1(state1)
    x1 = F.relu(x1)
    x1 = self.conv2(x1)
    x1 = F.relu(x1)
    x1 = F.max_pool2d(x1, 2)
    x1 = self.dropout1(x1)
    x1 = self.conv3(x1)
    x1 = self.avgpooling1(x1)
    x1 = torch.flatten(x1, 1)
    x1 = torch.cat([x1,state2, u], 1)
    x1 = self.fc1(x1)
    x1 = F.relu(x1)
    x1 = self.dropout2(x1)
    x1 = self.fc2(x1)
    return x1

class ReplayBuffer(object):
  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state.cpu(), copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

class TD3(object):
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action
    
    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        for it in range(iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            
            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state)
            
            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            
            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()
            
            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
            # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
    
    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))