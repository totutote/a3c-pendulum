import torch
import torch.nn as nn
import torch.nn.functional as F

class A3CModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(A3CModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def compute_loss(self, policy, rewards, values, next_values, dones, gamma=0.99):
        returns = rewards + (1 - dones) * gamma * next_values
        advantage = returns - values
        policy_loss = -torch.mean(torch.log(policy) * advantage.detach())
        value_loss = F.mse_loss(values, returns)
        return policy_loss + value_loss

    def backward(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()