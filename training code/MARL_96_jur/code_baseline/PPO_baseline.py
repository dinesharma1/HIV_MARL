# %%
import numpy as np
import math 
import random
import os
import glob
import time
from datetime import datetime
import timeit

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

# %%
# set device to cpu or cuda
device = torch.device('cpu')

#"""
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
#"""
    
print("============================================================================================")

# %%
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.actions_others = []
        self.obs_states = []
        self.full_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.new_inf = [] #sonza changes
        self.penalty = []
    

    def clear(self):
        del self.actions[:]
        del self.actions_others[:]
        del self.obs_states[:]
        del self.full_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.new_inf[:] #sonza changes
        del self.penalty[:]

# %%
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(obs_dim, 64),
                            nn.ReLU(),
                            nn.Linear(64, 64),
                            nn.ReLU(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(obs_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(obs_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, obs_state):

        if self.has_continuous_action_space:
            action_mean = self.actor(obs_state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(obs_state)
            dist = Categorical(action_probs)

        action = dist.sample()
        # #clamp the action 
        # action = torch.clamp(action, min=0, max=1)
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, obs_state, full_state, action, action_other):

        if self.has_continuous_action_space:
            #print('state')
            #print(state)
            #print(obs_state.size())
            action_mean = self.actor(obs_state) #buffer x 15
            #print("action mean")
            #print(action_mean.size()) #buffer x 9
            #print('Action evaluate')
            #print(action_mean)
            # print(self.action_var)
            action_var = self.action_var.expand_as(action_mean)
            # print('action variance', action_var)
            cov_mat = torch.diag_embed(action_var).to(device)
            # print('covaraince', cov_mat)
            # print(action_mean)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(obs_state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(obs_state)
        
        return action_logprobs, state_values, dist_entropy

# %%
class PPO:
    def __init__(self, obs_dim, state_dim, action_dim, lr_actor, lr_critic, gamma_, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma_ = gamma_
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(obs_dim, state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(obs_dim, state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                # print("setting actor output action_std to min_action_std : ", self.action_std)
            # else:
                # print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        # print("--------------------------------------------------------------------------------------------")


    def select_action(self, obs_state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                obs_state = torch.FloatTensor(obs_state).to(device)
                action, action_logprob = self.policy_old.act(obs_state)

            self.buffer.obs_states.append(obs_state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                obs_state = torch.FloatTensor(obs_state).to(device)
                action, action_logprob = self.policy_old.act(obs_state)
            
            self.buffer.obs_states.append(obs_state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma_ * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
#         print('reward before normalizing')
#         print(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
#         print('rewards after normalizing')
#         print(rewards)

        # convert list to tensor
        #print("sonzatest", len(self.buffer.obs_states)) #120
        old_obs_states = torch.squeeze(torch.stack(self.buffer.obs_states, dim=0)).detach().to(device)
        #print("sonzatest",old_obs_states.size()) #120x15
        old_full_states = torch.squeeze(torch.stack(self.buffer.full_states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        old_actions_others = 0 #only used for ctde_a, so setting it to dummy value and ignoring as not used

        #old/og
        #old_actions_others = torch.squeeze(torch.stack(self.buffer.actions_others, dim=0)).detach().to(device)

        # batch mean of important numbers
        loss_mean = 0
        ppo_objective_mean = 0
        value_function_loss_mean = 0
        entropy_mean = 0

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            #t1 = timeit.default_timer()
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_obs_states, old_full_states, old_actions, old_actions_others)
            #t2 = timeit.default_timer()
            #print("Policy Eval Time Taken  : ", t2-t1)
            

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            ppo_obj = torch.min(surr1, surr2)
            vf_loss = self.MseLoss(state_values, rewards)
            loss = -ppo_obj + 0.5*vf_loss - 0.01*dist_entropy
            #print("loss",loss)

            loss_mean += loss.mean()
            ppo_objective_mean += torch.min(surr1, surr2).mean()
            value_function_loss_mean += vf_loss
            entropy_mean += dist_entropy.mean()

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
#             dot = get_dot()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        #objectives = [loss_mean, ppo_objective_mean, value_function_loss_mean, entropy_mean]
        #return [obj.detach().numpy() / (self.K_epochs) for obj in objectives]        # avg over minibatches
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))



