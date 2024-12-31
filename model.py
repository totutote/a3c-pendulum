import torch
import torch.nn as nn
import torch.nn.functional as F

class A3CModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A3CModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #policy = torch.tanh(self.policy_head(x))
        policy = self.policy_head(x)
        print(policy)
        value = self.value_head(x)
        return policy, value

    def compute_loss(self, policy, rewards, values, next_values, dones):
        returns = rewards + (1 - dones) * next_values
        advantage = returns - values
        policy_loss = -torch.mean(torch.log(policy) * advantage.detach())
        value_loss = F.mse_loss(values, returns)
        return policy_loss + value_loss

    def backward(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()