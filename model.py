import torch
import torch.nn as nn
import torch.nn.functional as F

class A3CModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=512):
        super(A3CModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        actor_mean = self.actor_mean(x)
        actor_logstd = self.actor_logstd.expand_as(actor_mean)  # バッチサイズに合わせる
        actor_logstd = torch.clamp(actor_logstd, min=-20, max=2)  # ログ標準偏差をクランプ
        critic = self.critic(x)
        return actor_mean, actor_logstd, critic

    def compute_loss(self, dist, actions, rewards, critic_value, next_critic_value, dones, gamma):
        # 行動の確率分布を計算
        action_log_probs = dist.log_prob(actions).sum(dim=-1) # 各次元のlog_probを足し合わせる

        # Criticの損失を計算
        td_target = rewards + gamma * next_critic_value * (1 - dones)
        critic_loss = F.mse_loss(critic_value, td_target.detach()) # テンソルサイズを一致させるためsqueezeを削除

        # アドバンテージを計算
        advantage = (td_target - critic_value.squeeze()).detach()

        # Actorの損失を計算
        actor_loss = -(action_log_probs * advantage).mean()

        # エントロピーボーナス
        entropy = dist.entropy().mean()
        entropy_coef = 0.5 # エントロピーの係数。ハイパーパラメータ

        loss = actor_loss + critic_loss - entropy_coef * entropy
        return loss