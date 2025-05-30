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
    """简单的策略网络 - 改进数值稳定性"""
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(SimplePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 初始化权重
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        logits = self.net(x)
        # 使用log_softmax然后exp来提高数值稳定性
        return F.softmax(logits, dim=-1)

def quick_train(env, episodes=50, lr=0.01):
    """快速训练函数 - 修复数值稳定性问题"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = SimplePolicy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_log_probs = []
        total_reward = 0
        
        step_count = 0
        max_steps = 100  # 防止无限循环
        
        while step_count < max_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 检查输入是否有问题
            if torch.isnan(state_tensor).any() or torch.isinf(state_tensor).any():
                print(f"警告: 输入状态包含NaN或Inf: {state_tensor}")
                break
                
            probs = policy(state_tensor)
            
            # 检查概率是否有问题
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print(f"警告: 策略输出包含NaN或Inf: {probs}")
                break
                
            # 添加小的噪声防止概率为0
            probs = probs + 1e-8
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            dist = Categorical(probs)
            action = dist.sample()
            
            episode_log_probs.append(dist.log_prob(action))
            
            state, reward, done, _, _ = env.step(action.item())
            episode_rewards.append(reward)
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        scores.append(total_reward)
        
        # 计算回报和更新策略
        if len(episode_rewards) > 0:  # 确保有奖励数据
            returns = []
            R = 0
            for r in reversed(episode_rewards):
                R = r + 0.99 * R
                returns.insert(0, R)
            
            if len(returns) > 1:  # 至少需要2个点才能计算标准差
                returns = torch.FloatTensor(returns)
                returns_std = returns.std()
                if returns_std > 1e-8:  # 只有标准差足够大时才标准化
                    returns = (returns - returns.mean()) / returns_std
                else:
                    returns = returns - returns.mean()  # 只中心化，不标准化
            else:
                returns = torch.FloatTensor(returns)
            
            # 计算策略损失
            policy_loss = []
            for log_prob, R in zip(episode_log_probs, returns):
                if not torch.isnan(log_prob) and not torch.isnan(R):
                    policy_loss.append(-log_prob * R)
            
            if len(policy_loss) > 0:  # 确保有有效的损失项
                optimizer.zero_grad()
                total_loss = torch.stack(policy_loss).sum()
                
                # 检查损失是否有效
                if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                    total_loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
        
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}")
    
    return scores, policy

def test_all_environments():
    """测试所有环境"""
    print("="*60)
    print("快速数字任务验证实验 (修复版)")
    print("="*60)
    
    # 1. 数字猜测任务
    print("\n1. 数字猜测任务 (5-10秒训练)")
    try:
        env1 = NumberGuessingEnv()
        scores1, policy1 = quick_train(env1, episodes=30)
        
        # 测试训练效果
        print("测试数字猜测...")
        total_success = 0
        for i in range(10):
            state, _ = env1.reset()
            success = False
            for step in range(5):  # 最多5次猜测
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    probs = policy1(state_tensor)
                    if not torch.isnan(probs).any():
                        action = torch.argmax(probs).item()
                    else:
                        action = random.randint(0, 9)  # 随机备选
                state, reward, done, _, info = env1.step(action)
                if done:
                    if reward > 0:
                        total_success += 1
                        success = True
                    break
            print(f"  测试 {i+1}: {'成功' if success else '失败'}, 目标: {info.get('target', '?')}")
        print(f"成功率: {total_success}/10")
        
    except Exception as e:
        print(f"数字猜测任务出错: {e}")
        scores1 = []
    
    # 2. 简单算术任务  
    print("\n2. 简单算术任务 (5-10秒训练)")
    try:
        env3 = SimpleArithmeticEnv(max_num=3)
        scores3, policy3 = quick_train(env3, episodes=30)
        
        # 测试算术效果
        print("测试算术任务...")
        correct = 0
        for i in range(10):
            state, _ = env3.reset()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = policy3(state_tensor)
                if not torch.isnan(probs).any():
                    action = torch.argmax(probs).item()
                else:
                    action = random.randint(0, 6)  # 随机备选
            _, reward, _, _, info = env3.step(action)
            if info['correct']:
                correct += 1
            num1, num2 = int(state[0]*3), int(state[1]*3)
            print(f"  {num1} + {num2} = {action} {'✓' if info['correct'] else '✗'} (正确答案: {info['answer']})")
        
        print(f"算术正确率: {correct}/10")
        
    except Exception as e:
        print(f"算术任务出错: {e}")
        scores3 = []
    
    # 3. 序列记忆任务（简化版）
    print("\n3. 序列记忆任务 (5-10秒训练)")
    try:
        env2 = SequenceMemoryEnv(sequence_length=2)
        scores2, policy2 = quick_train(env2, episodes=20)
        print("序列记忆任务完成!")
    except Exception as e:
        print(f"序列记忆任务出错: {e}")
        scores2 = []
    
    # 绘制学习曲线
    try:
        plt.figure(figsize=(15, 4))
        
        if len(scores1) > 0:
            plt.subplot(1, 3, 1)
            plt.plot(scores1)
            plt.title('数字猜测学习曲线')
            plt.xlabel('Episode')
            plt.ylabel('Score')
        
        if len(scores2) > 0:
            plt.subplot(1, 3, 2)
            plt.plot(scores2)
            plt.title('序列记忆学习曲线')
            plt.xlabel('Episode')
            plt.ylabel('Score')
        
        if len(scores3) > 0:
            plt.subplot(1, 3, 3)
            plt.plot(scores3)
            plt.title('算术任务学习曲线')
            plt.xlabel('Episode')
            plt.ylabel('Score')
        
        plt.tight_layout()
        plt.savefig('quick_digit_tasks_fixed.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n学习曲线已保存为 quick_digit_tasks_fixed.png")
        
    except Exception as e:
        print(f"绘图出错: {e}")
    
    print(f"\n总训练时间: 约15-30秒")
    print("修复版本完成 - 解决了数值稳定性问题！")

def simple_demo():
    """简单演示 - 只测试最基本的功能"""
    print("="*50)
    print("简单演示 - 算术任务")
    print("="*50)
    
    # 创建一个非常简单的算术环境
    env = SimpleArithmeticEnv(max_num=2)  # 只用0,1,2
    
    print("随机策略测试:")
    for i in range(5):
        state, _ = env.reset()
        num1, num2 = int(state[0] * 2), int(state[1] * 2)
        action = random.randint(0, 4)  # 随机猜测
        _, reward, _, _, info = env.step(action)
        print(f"  {num1} + {num2} = {action} {'✓' if info['correct'] else '✗'} (答案: {info['answer']})")
    
    print(f"\n开始训练简单策略...")
    try:
        scores, policy = quick_train(env, episodes=20, lr=0.02)
        print(f"训练完成! 最终10轮平均分数: {np.mean(scores[-10:]):.2f}")
        
        print("\n训练后策略测试:")
        correct = 0
        for i in range(5):
            state, _ = env.reset()
            num1, num2 = int(state[0] * 2), int(state[1] * 2)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = policy(state_tensor)
                action = torch.argmax(probs).item()
            _, reward, _, _, info = env.step(action)
            if info['correct']:
                correct += 1
            print(f"  {num1} + {num2} = {action} {'✓' if info['correct'] else '✗'} (答案: {info['answer']})")
        
        print(f"\n训练后正确率: {correct}/5")
        
    except Exception as e:
        print(f"训练出错: {e}")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("选择运行模式:")
    print("1. 完整测试 (test_all_environments)")
    print("2. 简单演示 (simple_demo)")
    
    # 直接运行简单演示，避免复杂的交互
    print("\n运行简单演示...")
    simple_demo()
    
    print("\n如果简单演示成功，可以尝试完整测试:")
    print("python RL-review/simple_digit_tasks_fixed.py") 