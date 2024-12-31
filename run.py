import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
from model import A3CModel

def run(model_path, video_folder='videos'):
    # Pendulum環境の初期化
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: x == 0)
    model = A3CModel(env.observation_space.shape[0], env.action_space.shape[0])
    
    # 学習済みモデルのロード
    model.load_state_dict(torch.load(model_path))
    model.eval()

    state, _ = env.reset()
    terminated, truncated = False, False

    while not (terminated or truncated):
        # 状態をテンソルに変換
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # アクションの選択
        action, _ = model.forward(state_tensor)
        action = action.detach().numpy()[0]
        
        # 環境でアクションを実行
        state, reward, terminated, truncated, _ = env.step(action)
        env.render()

    env.close()

if __name__ == "__main__":
    run('model.pth')  # 学習済みモデルのパスを指定