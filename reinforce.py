import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 存储轨迹数据
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def update(self):
        R = 0
        policy_loss = []
        returns = deque()
        
        # 计算回报
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化
        
        # 计算策略损失
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # 更新网络
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # 清空轨迹数据
        del self.rewards[:]
        del self.saved_log_probs[:]

def train_reinforce(env_name='CartPole-v1', episodes=1000, print_every=100):
    """
    训练REINFORCE算法的主函数
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCE(state_dim, action_dim)
    
    scores = []
    recent_scores = deque(maxlen=100)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        scores.append(total_reward)
        recent_scores.append(total_reward)
        
        # 更新策略
        agent.update()
        
        # 打印进度
        if episode % print_every == 0:
            avg_score = np.mean(recent_scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
            
            # 如果最近100轮平均分数超过195，认为任务解决
            if avg_score >= 195.0:
                print(f"Environment solved in {episode} episodes!")
                break
    
    env.close()
    return scores, agent

def evaluate_agent(agent, env_name='CartPole-v1', num_episodes=10):
    """
    评估训练好的智能体
    """
    env = gym.make(env_name, render_mode='human')
    
    total_scores = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            # 使用确定性策略（选择概率最高的动作）
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                probs = agent.policy_net(state_tensor)
                action = torch.argmax(probs).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        total_scores.append(total_reward)
        print(f"Episode {episode + 1}: Score = {total_reward}")
    
    env.close()
    avg_score = np.mean(total_scores)
    print(f"\nAverage Score over {num_episodes} episodes: {avg_score:.2f}")
    return total_scores

def plot_scores(scores, title="REINFORCE Training Progress"):
    """
    绘制训练过程中的分数变化
    """
    plt.figure(figsize=(12, 4))
    
    # 原始分数
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # 移动平均
    plt.subplot(1, 2, 2)
    moving_avg = []
    window = 100
    for i in range(len(scores)):
        if i < window:
            moving_avg.append(np.mean(scores[:i+1]))
        else:
            moving_avg.append(np.mean(scores[i-window+1:i+1]))
    
    plt.plot(moving_avg)
    plt.title(f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.axhline(y=195, color='r', linestyle='--', label='Solved threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('reinforce_training.png')
    plt.show()

def run_experiments():
    """
    运行完整的实验流程
    """
    print("="*50)
    print("REINFORCE算法验证实验")
    print("="*50)
    
    # 1. 训练阶段
    print("\n1. 开始训练REINFORCE算法...")
    scores, trained_agent = train_reinforce(episodes=1000)
    
    # 2. 绘制训练曲线
    print("\n2. 绘制训练曲线...")
    plot_scores(scores)
    
    # 3. 保存模型
    print("\n3. 保存训练好的模型...")
    torch.save(trained_agent.policy_net.state_dict(), 'reinforce_cartpole.pth')
    
    # 4. 评估阶段
    print("\n4. 评估训练好的智能体...")
    eval_scores = evaluate_agent(trained_agent, num_episodes=5)
    
    # 5. 统计结果
    print("\n5. 实验结果统计:")
    print(f"   训练集总episode数: {len(scores)}")
    print(f"   最终100轮平均分数: {np.mean(scores[-100:]):.2f}")
    print(f"   评估集平均分数: {np.mean(eval_scores):.2f}")
    print(f"   评估集标准差: {np.std(eval_scores):.2f}")
    
    return scores, eval_scores, trained_agent

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 运行实验
    try:
        scores, eval_scores, agent = run_experiments()
        print("\n实验完成！")
    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        print("请确保已安装必要的依赖: pip install torch gymnasium matplotlib")
