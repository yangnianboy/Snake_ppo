import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import PPOConfig


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_layers=3):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Sequential(nn.Linear(hidden_size, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()
    
    def evaluate(self, states, actions):
        probs, values = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), values.squeeze(), dist.entropy()


class RolloutBuffer:
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones = [], [], []
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, last_value, gamma, gae_lambda):
        values = np.array(self.values + [last_value])
        rewards, dones = np.array(self.rewards), np.array(self.dones)
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        return values[:-1] + advantages, advantages
    
    def get_batches(self, batch_size, returns, advantages):
        indices = np.random.permutation(len(self.states))
        for start in range(0, len(self.states), batch_size):
            idx = indices[start:start + batch_size]
            yield (torch.FloatTensor(np.array(self.states)[idx]),
                   torch.LongTensor(np.array(self.actions)[idx]),
                   torch.FloatTensor(np.array(self.log_probs)[idx]),
                   torch.FloatTensor(returns[idx]),
                   torch.FloatTensor(advantages[idx]))


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.cfg = PPOConfig()
        self.model = ActorCritic(state_dim, action_dim, self.cfg.HIDDEN_SIZE, self.cfg.NUM_LAYERS)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.cfg.LR_DECAY)
        self.buffer = RolloutBuffer()
    
    def get_action(self, state):
        return self.model.get_action(state)
    
    def update(self):
        with torch.no_grad():
            _, last_value = self.model(torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0))
            last_value = last_value.item() if not self.buffer.dones[-1] else 0
        
        returns, advantages = self.buffer.compute_gae(last_value, self.cfg.GAMMA, self.cfg.GAE_LAMBDA)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.cfg.NUM_EPOCHS):
            for states, actions, old_lp, ret, adv in self.buffer.get_batches(self.cfg.BATCH_SIZE, returns, advantages):
                new_lp, values, entropy = self.model.evaluate(states, actions)
                ratio = torch.exp(new_lp - old_lp)
                surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - self.cfg.CLIP_EPSILON, 1 + self.cfg.CLIP_EPSILON) * adv)
                loss = -surr.mean() + self.cfg.VALUE_COEF * nn.MSELoss()(values, ret) - self.cfg.ENTROPY_COEF * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
                self.optimizer.step()
        
        self.buffer.clear()
        self.scheduler.step()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
