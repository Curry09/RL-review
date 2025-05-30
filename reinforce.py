import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque

class NumberGuessingEnv(gym.Env):
    """
    数字猜测环境：智能体需要猜测一个0-9之间的数字
    每次猜错会得到提示（太大或太小），猜对得到奖励
    """
    def __init__(self):
        super(NumberGuessingEnv, self).__init__()
        self.action_space = spaces.Discrete(10)  # 0-9
        self.observation_space = spaces.Box(low=0, high=2, shape=(3,), dtype=np.float32)
        
        self.target = None
        self.guesses = 0
        self.max_guesses = 5
        self.last_guess = -1
        self.last_hint = 0  # 0: 相等, 1: 太大, -1: 太小
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.target = random.randint(0, 9)
        self.guesses = 0
        self.last_guess = -1
        self.last_hint = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        # 状态：[剩余猜测次数/最大次数, 上次猜测/9, 提示(-1,0,1)]
        return np.array([
            (self.max_guesses - self.guesses) / self.max_guesses,
            self.last_guess / 9.0 if self.last_guess >= 0 else 0,
            (self.last_hint + 1) / 2.0  # 归一化到0-1
        ], dtype=np.float32)
    
    def step(self, action):
        self.guesses += 1
        self.last_guess = action
        
        if action == self.target:
            self.last_hint = 0
            reward = 10 - self.guesses  # 越早猜对奖励越高
            terminated = True
        elif action > self.target:
            self.last_hint = 1
            reward = -1
            terminated = False
        else:
            self.last_hint = -1
            reward = -1
            terminated = False
        
        if self.guesses >= self.max_guesses and not terminated:
            terminated = True
            reward = -5  # 超时惩罚
        
        return self._get_obs(), reward, terminated, False, {"target": self.target}

class SequenceMemoryEnv(gym.Env):
    """
    序列记忆环境：智能体需要记住并重复一个数字序列
    """
    def __init__(self, sequence_length=3):
        super(SequenceMemoryEnv, self).__init__()
        self.sequence_length = sequence_length
        self.action_space = spaces.Discrete(4)  # 0,1,2,3四个数字
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        
        self.target_sequence = []
        self.current_step = 0
        self.show_phase = True  # True: 显示序列, False: 回答阶段
        self.phase_step = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.target_sequence = [random.randint(0, 3) for _ in range(self.sequence_length)]
        self.current_step = 0
        self.show_phase = True
        self.phase_step = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        obs = np.zeros(8, dtype=np.float32)
        
        # 前4位：当前显示的数字（one-hot编码）
        if self.show_phase and self.phase_step < len(self.target_sequence):
            obs[self.target_sequence[self.phase_step]] = 1.0
        
        # 第5位：是否在显示阶段
        obs[4] = 1.0 if self.show_phase else 0.0
        
        # 第6位：当前步骤在序列中的位置
        obs[5] = self.phase_step / max(1, self.sequence_length)
        
        # 第7位：总进度
        obs[6] = self.current_step / (2 * self.sequence_length)
        
        # 第8位：剩余步骤
        obs[7] = (2 * self.sequence_length - self.current_step) / (2 * self.sequence_length)
        
        return obs
    
    def step(self, action):
        self.current_step += 1
        
        if self.show_phase:
            # 显示阶段，动作被忽略
            self.phase_step += 1
            if self.phase_step >= self.sequence_length:
                self.show_phase = False
                self.phase_step = 0
            return self._get_obs(), 0, False, False, {}
        else:
            # 回答阶段
            if action == self.target_sequence[self.phase_step]:
                reward = 2
                self.phase_step += 1
                if self.phase_step >= self.sequence_length:
                    # 完成整个序列
                    return self._get_obs(), reward + 5, True, False, {"success": True}
            else:
                reward = -2
                return self._get_obs(), reward, True, False, {"success": False}
            
            return self._get_obs(), reward, False, False, {}

class SimpleArithmeticEnv(gym.Env):
    """
    简单算术环境：给出两个数字，智能体需要计算它们的和
    """
    def __init__(self, max_num=5):
        super(SimpleArithmeticEnv, self).__init__()
        self.max_num = max_num
        self.action_space = spaces.Discrete(2 * max_num + 1)  # 0 到 2*max_num
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        self.num1 = 0
        self.num2 = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.num1 = random.randint(0, self.max_num)
        self.num2 = random.randint(0, self.max_num)
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([
            self.num1 / self.max_num,
            self.num2 / self.max_num,
            (self.num1 + self.num2) / (2 * self.max_num),  # 提示
            1.0  # 偏置
        ], dtype=np.float32)
    
    def step(self, action):
        correct_answer = self.num1 + self.num2
        if action == correct_answer:
            reward = 10
        else:
            reward = -abs(action - correct_answer)  # 越接近越好
        
        return self._get_obs(), reward, True, False, {
            "correct": action == correct_answer,
            "answer": correct_answer,
            "guess": action
        }

class SimplePolicy(nn.Module):
    """简单的策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(SimplePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

def quick_train(env, episodes=50, lr=0.01):
    """快速训练函数"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = SimplePolicy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    scores = []
    log_probs = []
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_log_probs = []
        total_reward = 0
        
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            
            episode_log_probs.append(dist.log_prob(action))
            
            state, reward, done, _, _ = env.step(action.item())
            episode_rewards.append(reward)
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        
        # 计算回报和更新策略
        returns = []
        R = 0
        for r in reversed(episode_rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, R in zip(episode_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}")
    
    return scores, policy

def test_all_environments():
    """测试所有环境"""
    print("="*60)
    print("快速数字任务验证实验")
    print("="*60)
    
    # 1. 数字猜测任务
    print("\n1. 数字猜测任务 (5-10秒训练)")
    env1 = NumberGuessingEnv()
    scores1, policy1 = quick_train(env1, episodes=30)
    
    # 测试训练效果
    print("测试数字猜测...")
    total_success = 0
    for _ in range(10):
        state, _ = env1.reset()
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = policy1(state_tensor)
                action = torch.argmax(probs).item()
            state, reward, done, _, info = env1.step(action)
            if done:
                if reward > 0:
                    total_success += 1
                break
    print(f"成功率: {total_success}/10")
    
    # 2. 序列记忆任务
    print("\n2. 序列记忆任务 (5-10秒训练)")
    env2 = SequenceMemoryEnv(sequence_length=2)
    scores2, policy2 = quick_train(env2, episodes=40)
    
    # 3. 简单算术任务
    print("\n3. 简单算术任务 (5-10秒训练)")
    env3 = SimpleArithmeticEnv(max_num=3)
    scores3, policy3 = quick_train(env3, episodes=30)
    
    # 测试算术效果
    print("测试算术任务...")
    correct = 0
    for _ in range(20):
        state, _ = env3.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = policy3(state_tensor)
            action = torch.argmax(probs).item()
        _, reward, _, _, info = env3.step(action)
        if info['correct']:
            correct += 1
        print(f"  {int(state[0]*3)} + {int(state[1]*3)} = {action} {'✓' if info['correct'] else '✗'}")
    
    print(f"算术正确率: {correct}/20")
    
    # 绘制学习曲线
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(scores1)
    plt.title('数字猜测学习曲线')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 3, 2)
    plt.plot(scores2)
    plt.title('序列记忆学习曲线')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 3, 3)
    plt.plot(scores3)
    plt.title('算术任务学习曲线')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('quick_digit_tasks.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n总训练时间: 约15-30秒")
    print("所有任务都是专门设计的快速验证环境！")

def interactive_demo():
    """交互式演示"""
    print("\n" + "="*40)
    print("交互式演示")
    print("="*40)
    
    # 让用户选择环境
    print("选择要演示的环境:")
    print("1. 数字猜测")
    print("2. 序列记忆")
    print("3. 简单算术")
    
    try:
        choice = int(input("请输入选择 (1-3): "))
        
        if choice == 1:
            env = NumberGuessingEnv()
            print("开始数字猜测演示...")
            state, _ = env.reset()
            print(f"我想了一个0-9的数字，你来猜！")
            
            while True:
                print(f"状态: {state}")
                action = int(input("你的猜测 (0-9): "))
                state, reward, done, _, info = env.step(action)
                print(f"奖励: {reward}")
                if done:
                    if reward > 0:
                        print(f"恭喜！答案是 {info['target']}")
                    else:
                        print(f"游戏结束！答案是 {info['target']}")
                    break
                elif state[2] > 0.5:  # 太大
                    print("太大了！")
                elif state[2] < 0.5:  # 太小
                    print("太小了！")
                    
        elif choice == 3:
            env = SimpleArithmeticEnv(max_num=5)
            print("开始算术演示...")
            state, _ = env.reset()
            num1, num2 = int(state[0] * 5), int(state[1] * 5)
            print(f"计算: {num1} + {num2} = ?")
            answer = int(input("你的答案: "))
            _, reward, _, _, info = env.step(answer)
            print(f"正确答案: {info['answer']}, 你的答案: {info['guess']}")
            print(f"正确性: {'对了！' if info['correct'] else '错了！'}")
            
    except:
        print("输入无效，跳过交互演示")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 运行所有测试
    test_all_environments()
    
    # 交互式演示
    interactive_demo() 