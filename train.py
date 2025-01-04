import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from model import A3CModel
import shutil
import os
import matplotlib.pyplot as plt
import numpy as np

def worker(
    global_model,
    optimizer,
    env_name,
    num_episodes,
    worker_id,
    conn,
    video_folder="videos",
    video_interval=100,
    gamma=0.95,
):
    if worker_id == 0:
        env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda x: x % video_interval == 0,
        )
    else:
        env = gym.make(env_name)

    epsilon = 1.0  # 初期値
    epsilon_decay = 0.9 # εの減衰率
    min_epsilon = 0.0  # εの最小値

    local_model = A3CModel(env.observation_space.shape[0], env.action_space.shape[0])
    local_model.load_state_dict(global_model.state_dict())
    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0
        accumulated_loss = 0

        while not (terminated or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # 1. 状態を入力し、Actorの出力を取得
            actor_mean, actor_logstd, critic_value = local_model(state_tensor)

            # 2. 確率分布を生成
            dist = torch.distributions.Normal(actor_mean, torch.exp(actor_logstd))

            # 3. epsilon-greedyによる行動選択
            if np.random.rand() < epsilon:
                action = torch.FloatTensor(np.array([env.action_space.sample()]))
            else:
                action = dist.sample()

            # 環境と相互作用
            next_state, reward, terminated, truncated, _ = env.step(action.detach().numpy()[0])
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32))
            done_tensor = torch.tensor([1.0 if terminated or truncated else 0.0], dtype=torch.float32)

            _, _, next_critic_value = local_model(next_state_tensor)

            total_reward += reward.item()

             # 5. 損失を計算
            loss = local_model.compute_loss(dist, action, reward, critic_value, next_critic_value, done_tensor, gamma)
            loss.backward()
            accumulated_loss += loss.item()

            state = next_state

        with torch.no_grad():
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                if global_param.grad is None:
                    global_param.grad = local_param.grad.clone()
                else:
                    global_param.grad += local_param.grad
        optimizer.step()
        optimizer.zero_grad()
        local_model.load_state_dict(global_model.state_dict())

        # epsilonの減衰
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        conn.send((worker_id, episode + 1, total_reward, epsilon, accumulated_loss))
    conn.close()
    env.close()


def train(
    env_name="Pendulum-v1",
    num_episodes=10000,
    learning_rate=0.02,
    num_workers=24,
    video_folder="videos",
    video_interval=50,
):
    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)

    env = gym.make(env_name)
    global_model = A3CModel(env.observation_space.shape[0], env.action_space.shape[0])
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)
    processes = []
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(num_workers)])
    rewards = []  # 報酬を記録するリスト

    plt.ion()  # インタラクティブモードを有効にする
    fig, ax = plt.subplots()
    line, = ax.plot(rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward per Episode')

    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(
                global_model,
                optimizer,
                env_name,
                num_episodes,
                worker_id,
                child_conns[worker_id],
                video_folder,
                video_interval,
            ),
        )
        p.start()
        processes.append(p)

    for episode in range(num_episodes):
        for parent_conn in parent_conns:
            worker_id, ep, total_reward, epsilon, accumulated_loss = parent_conn.recv()
            rewards.append(total_reward)  # 報酬を記録
            print(
                f"Worker {worker_id}, Episode {ep}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f},  Accumulated Loss: {accumulated_loss:.2f}"
            )

            # グラフを更新
            line.set_ydata(rewards)
            line.set_xdata([i / num_workers for i in range(len(rewards))])
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

    for p in processes:
        p.join()

    plt.ioff()
    plt.savefig('reward.png')
    plt.show()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    train()
