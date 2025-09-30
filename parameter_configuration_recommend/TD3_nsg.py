# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optimizer
from prioritized_replay_memory import PrioritizedReplayMemory
from utils import Scaler_para_gpu_nsg

class OUProcess():
    def __init__(self, num_data, size, sigma_decay_rate, n_steps_annealing, sigma, mu=0.0, sigma_min = 0.002, theta=0.3, dt=1e-2, x_prev=None):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_decay_rate = sigma_decay_rate
        self.dt = dt
        self.x_prev = x_prev
        self.shape = (num_data, size)
        self.n_steps_annealing = n_steps_annealing

        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.0
            self.c = sigma
            self.sigma_min = sigma

        self.reset(sigma=sigma)

    def reset(self, sigma):
        self.x_prev = self.x_prev if self.x_prev is not None else np.zeros(self.shape)
        self.n_steps = 0
        self.sigma = sigma

    def current_sigma(self):
        sigma = max(self.sigma_min, self.sigma * (self.sigma_decay_rate ** self.n_steps))

        return sigma

    def noise(self):
        current_sigma = self.current_sigma()

        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.shape)
        self.x_prev = x

        self.n_steps += 1
    
        return x

class Gaussian_noise_action():
    def __init__(self, num_data, size, sigma_decay_rate, sigma, sigma_min=0.001):
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay_rate = sigma_decay_rate

        self.shape = (num_data, size)

        self.n_steps = 0

        self.reset(sigma=sigma)

    def reset(self, sigma):
        self.n_steps = 0
        self.sigma = sigma

    def noise(self):
        current_sigma = max(self.sigma_min, self.sigma * (self.sigma_decay_rate ** self.n_steps))

        x = np.random.normal(loc=0.0, scale=current_sigma, size=self.shape)

        self.n_steps += 1

        return x

class Gaussian_noise_value():
    def __init__(self, num_data, size, sigma):
        self.sigma = sigma
        self.shape = (num_data, size)
        self.n_steps = 0

    def noise(self):
        x = np.random.normal(loc=0.0, scale=self.sigma, size=self.shape)
        return x

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, actor_layer_sizes, dv):
        super(Actor, self).__init__()
        layer_list = []
        layer_sizes = [int(x) for x in actor_layer_sizes]

        layer_list.append(nn.Linear(n_states, layer_sizes[0]))
        layer_list.append(nn.ReLU(inplace=False))

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))
            layer_list.append(nn.ReLU(inplace=False))

        self.out = nn.Linear(layer_sizes[-1], n_actions)

        self.layers = nn.Sequential(*layer_list)
        self._init_weights()

        self.sigmoid = nn.Sigmoid()

    def _init_weights(self):
        for m in self.layers:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                nn.init.uniform_(m.bias, -0.1, 0.1)

        nn.init.normal_(self.out.weight, mean=0.0, std=1e-2)
        nn.init.uniform_(self.out.bias, -0.1, 0.1)

    def forward(self, x):
        out = self.sigmoid(self.out(self.layers(x)))
        return out

class Critic(nn.Module):
    def __init__(self, n_states, n_actions, critic_layer_sizes):
        super(Critic, self).__init__()
        layer_list = []
        layer_sizes = [int(x) for x in critic_layer_sizes]

        self.state_input = nn.Linear(n_states, 128)
        self.action_input = nn.Linear(n_actions, 128)

        for i in range(len(layer_sizes) - 1):  #
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))
            layer_list.append(nn.ReLU(inplace=False))

        layer_list.append(nn.Linear(layer_sizes[-1], 1))

        self.layers = nn.Sequential(*layer_list)
        self._init_weights()

        self.relu = nn.ReLU(inplace=False)

    def _init_weights(self):
        nn.init.normal_(self.state_input.weight, mean=0.0, std=1e-2)
        nn.init.uniform_(self.state_input.bias, -0.1, 0.1)

        nn.init.normal_(self.action_input.weight, mean=0.0, std=1e-2)
        nn.init.uniform_(self.action_input.bias, -0.1, 0.1)

        for m in self.layers:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, state, action):
        state = self.relu(self.state_input(state))
        action = self.relu(self.action_input(action))

        _input = torch.cat([state, action], dim=1)
        value = self.layers(_input)
        return value


class TD3(object):
    def __init__(self, n_states, n_actions, num_data, opt, dv):
        self.device = dv

        self.para_scaler = Scaler_para_gpu_nsg(dv)

        self.n_states = n_states
        self.n_actions = n_actions
        self.num_data = num_data

        self.actor_layer_sizes = opt['actor_layer_sizes']
        self.critic_layer_sizes = opt['critic_layer_sizes']

        # Params
        self.alr = opt['alr']
        self.clr = opt['clr']
        self.batch_size = opt['batch_size']
        self.gamma = opt['gamma']
        self.tau = opt['tau']
        self.update_time = 0
        self.delay_time = opt['delay_time']
        self.actor_path = opt['actor_path']
        self.critic1_path = opt['critic1_path']
        self.critic2_path = opt['critic2_path']

        self._build_network()

        self.replay_memory = PrioritizedReplayMemory(capacity=opt['memory_size'])

        self.action_noise = Gaussian_noise_action(num_data, n_actions, opt['sigma_decay_rate'],  opt['sigma'])
        self.OUP_noise = OUProcess(num_data, n_actions, opt['sigma_decay_rate'], opt['max_steps'], opt['sigma'])

        print('TD3 Initialzed!')

    @staticmethod
    def totensor(x):
        return torch.tensor(x).to(torch.float32)

    def _build_network(self):
        self.actor = Actor(self.n_states, self.n_actions, self.actor_layer_sizes, self.device)
        self.target_actor = Actor(self.n_states, self.n_actions, self.actor_layer_sizes, self.device)

        self.critic1 = Critic(self.n_states, self.n_actions, self.critic_layer_sizes)
        self.target_critic1 = Critic(self.n_states, self.n_actions, self.critic_layer_sizes)

        self.critic2 = Critic(self.n_states, self.n_actions, self.critic_layer_sizes)
        self.target_critic2 = Critic(self.n_states, self.n_actions, self.critic_layer_sizes)

        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.critic1.to(self.device)
        self.target_critic1.to(self.device)
        self.critic2.to(self.device)
        self.target_critic2.to(self.device)

        self._update_target(self.target_actor, self.actor, tau=1.0)
        self._update_target(self.target_critic1, self.critic1, tau=1.0)
        self._update_target(self.target_critic2, self.critic2, tau=1.0)

        self.loss_criterion = nn.MSELoss(reduction='mean')

        self.actor_optimizer = optimizer.Adam(lr=self.alr, params=self.actor.parameters(), weight_decay=1e-5)
        self.critic1_optimizer = optimizer.Adam(lr=self.clr, params=self.critic1.parameters(), weight_decay=1e-5)
        self.critic2_optimizer = optimizer.Adam(lr=self.clr, params=self.critic2.parameters(), weight_decay=1e-5)

        self.actor_scheduler = optimizer.lr_scheduler.StepLR(self.actor_optimizer, step_size=5, gamma=0.998)
        self.critic1_scheduler = optimizer.lr_scheduler.StepLR(self.critic1_optimizer, step_size=5, gamma=0.998)
        self.critic2_scheduler = optimizer.lr_scheduler.StepLR(self.critic2_optimizer, step_size=5, gamma=0.998)

        if os.path.exists(self.actor_path):
            epoch = self.load_model()
            print("Loading model from file: {}".format(self.actor_path))

    @staticmethod
    def _update_target(target, source, tau):
        for (target_param, param) in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)

    def reset(self, sigma):
        self.action_noise.reset(sigma)

    def _sample_batch(self):
        batch, idx = self.replay_memory.sample(self.batch_size)
        states = list(map(lambda x: x[0], batch))
        next_states = list(map(lambda x: x[3], batch))
        actions = list(map(lambda x: x[1], batch))
        rewards = list(map(lambda x: x[2], batch))
        terminates = list(map(lambda x: x[4], batch))

        return idx, states, next_states, actions, rewards, terminates

    def add_sample(self, states, actions, rewards, next_states, terminates):
        self.critic1.eval()
        self.critic2.eval()
        self.actor.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()
        self.target_actor.eval()

        batch_states = self.totensor(states).to(self.device)
        batch_next_states = self.totensor(next_states).to(self.device)
        batch_actions = self.totensor(actions).to(self.device)
        batch_rewards = self.totensor(rewards).to(self.device)

        with torch.no_grad():
            current_values1 = self.critic1(batch_states, batch_actions)
            current_values2 = self.critic2(batch_states, batch_actions)
            target_next_actions = self.target_actor(batch_next_states)

            discounts = self.totensor([0 if x[0] else 1 for x in terminates]).to(self.device)

            target_next_values1 = discounts * self.target_critic1(batch_next_states, target_next_actions) * self.gamma
            target_next_values2 = discounts * self.target_critic2(batch_next_states, target_next_actions) * self.gamma
            target_next_values = torch.min(target_next_values1, target_next_values2)
            target_values = batch_rewards + target_next_values

            errors = torch.abs(current_values1 + current_values2 - 2 * target_values).cpu().numpy().astype(np.float32)
            errors = errors / 2

        self.target_actor.train()
        self.actor.train()
        self.critic1.train()
        self.target_critic1.train()
        self.critic2.train()
        self.target_critic2.train()

        batch = states.shape[0]

        for i in range(batch):
            error = errors[i][0]
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            terminate = terminates[i]

            self.replay_memory.add(error, (state, action, reward, next_state, terminate))  # 这里要改成用循环添加，因为输入是一个二维数组

    def update(self):
        """ Update the Actor and Critic with a batch data
        """
        idxs, states, next_states, actions, rewards, terminates = self._sample_batch()  # 读出来是一个列表，列表里的元素是一个一维数组
        batch_states = self.totensor(np.vstack(states)).to(self.device)  # 用np.vstack将这个列表转换为二维数组
        batch_next_states = self.totensor(np.vstack(next_states)).to(self.device)
        batch_actions = self.totensor(np.vstack(actions)).to(self.device)
        batch_rewards = self.totensor(np.vstack(rewards)).to(self.device)
        mask = [0 if x[0] else 1 for x in terminates]
        mask = self.totensor(mask).to(self.device)
        mask = mask.reshape((mask.shape[0], 1))

        with torch.no_grad():
            target_next_actions = self.target_actor(batch_next_states).detach()

            target_next_value1 = (mask * self.target_critic1(batch_next_states, target_next_actions) * self.gamma).detach()
            target_next_value2 = (mask * self.target_critic2(batch_next_states, target_next_actions) * self.gamma).detach()
            target_next_value = torch.min(target_next_value1, target_next_value2)
            target_value = batch_rewards +target_next_value

        current_value1 = self.critic1(batch_states, batch_actions)
        current_value2 = self.critic2(batch_states, batch_actions)

        # update prioritized memory
        error = torch.abs(current_value1 + current_value2 - 2 * target_value).detach().cpu().numpy()
        error = error / 2
        for i in range(self.batch_size):
            idx = idxs[i]
            self.replay_memory.update(idx, error[i][0])

        # Update Critic
        loss1 = self.loss_criterion(current_value1, target_value)
        loss2 = self.loss_criterion(current_value2, target_value)
        loss = loss1 + loss2

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Update Actor
        self.update_time += 1
        if (self.update_time % self.delay_time) != 0:
            return

        self.critic1.eval()
        self.critic2.eval()

        actions = self.actor(batch_states)
        q1 = self.critic1(batch_states, actions)
        q2 = self.critic2(batch_states, actions)
        q = torch.min(q1, q2)
        policy_loss = -torch.mean(q)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic1.train()
        self.critic2.train()

        self._update_target(self.target_critic1, self.critic1, tau=self.tau)
        self._update_target(self.target_critic2, self.critic2, tau=self.tau)
        self._update_target(self.target_actor, self.actor, tau=self.tau)

        self.update_time = 0
        return loss.item(), policy_loss.item()

    def choose_action(self, x, train=True):
        x = self.totensor(x).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            act = self.actor(x)

        self.actor.train()

        action = act.cpu().numpy()
        if train:
            action += self.OUP_noise.noise()

        action = np.clip(action, 0, 1)
        return action

    def save_model(self, epoch):
        actor_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
        }

        critic1_checkpoint = {
            'model_state_dict': self.critic1.state_dict(),
            'optimizer_state_dict': self.critic1_optimizer.state_dict(),
        }

        critic2_checkpoint = {
            'model_state_dict': self.critic2.state_dict(),
            'optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }

        torch.save(actor_checkpoint, self.actor_path)
        torch.save(critic1_checkpoint, self.critic1_path)
        torch.save(critic2_checkpoint, self.critic2_path)

    def load_model(self):
        actor_checkpoint = torch.load(self.actor_path)
        self.actor.load_state_dict(actor_checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        epoch = actor_checkpoint['epoch']

        critic1_checkpoint = torch.load(self.critic1_path)
        self.critic1.load_state_dict(critic1_checkpoint['model_state_dict'])
        self.critic1_optimizer.load_state_dict(critic1_checkpoint['optimizer_state_dict'])

        critic2_checkpoint = torch.load(self.critic2_path)
        self.critic2.load_state_dict(critic2_checkpoint['model_state_dict'])
        self.critic2_optimizer.load_state_dict(critic2_checkpoint['optimizer_state_dict'])

        return epoch

