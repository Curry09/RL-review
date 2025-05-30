# 强化学习算法验证实验

这个项目包含了多个用于快速验证强化学习算法的实验环境，特别适合教学和算法原型验证。

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行实验
```bash
# 运行经典CartPole实验（训练时间: 1-3分钟）
python reinforce.py

# 运行快速数字任务验证（训练时间: 15-30秒）
python simple_digit_tasks.py
```

## 📊 实验环境介绍

### 1. 经典控制任务 (`reinforce.py`)

**CartPole-v1环境**
- **目标**: 保持杆子平衡尽可能长时间
- **状态空间**: 4维连续 (位置, 速度, 角度, 角速度)
- **动作空间**: 2个离散动作 (左推, 右推)
- **成功标准**: 连续100轮平均分数 ≥ 195
- **训练时间**: 通常200-800轮episode
- **算法**: REINFORCE (策略梯度)

**特点**:
- ✅ 经典benchmark，结果可对比
- ✅ 完整的训练/评估/可视化流程
- ✅ 模型保存与加载
- ❌ 训练时间较长（几分钟）

### 2. 快速数字任务 (`simple_digit_tasks.py`)

我们设计了三个超快速的验证环境：

#### 2.1 数字猜测环境 (NumberGuessingEnv)
- **目标**: 猜测0-9之间的随机数字
- **状态**: [剩余次数, 上次猜测, 提示信息]
- **动作**: 0-9的数字选择
- **奖励**: 猜对得分，越早猜对分数越高
- **训练时间**: ⚡ 5-10秒 (30轮episode)

```python
# 状态空间设计
状态 = [
    剩余猜测次数/最大次数,    # 0-1
    上次猜测/9,              # 0-1  
    提示信息                 # 0(太小), 0.5(相等), 1(太大)
]
```

#### 2.2 序列记忆环境 (SequenceMemoryEnv)
- **目标**: 记住并重复数字序列 (如: 2-1-3)
- **阶段**: 显示阶段 → 回答阶段
- **状态**: 8维，包含当前数字、阶段信息、进度等
- **动作**: 选择0-3的数字
- **训练时间**: ⚡ 5-10秒 (40轮episode)

#### 2.3 简单算术环境 (SimpleArithmeticEnv)
- **目标**: 计算两个小数字的和 (如: 2+3=5)
- **状态**: [数字1, 数字2, 答案提示, 偏置]
- **动作**: 选择计算结果
- **奖励**: 答对+10分，答错按距离扣分
- **训练时间**: ⚡ 5-10秒 (30轮episode)

## 🔬 验证实验的价值

### 为什么需要这些快速任务？

1. **算法正确性验证**: 
   - 快速检查REINFORCE实现是否有bug
   - 验证策略梯度计算是否正确
   - 确认网络结构设计合理

2. **超参数调试**:
   - 学习率选择 (0.001-0.01)
   - 网络结构调整
   - 奖励设计验证

3. **教学演示**:
   - 实时观察学习过程
   - 理解策略梯度原理
   - 可视化训练曲线

4. **新想法快速测试**:
   - 新的网络架构
   - 改进的损失函数
   - 不同的探索策略

## 📈 实验结果示例

### 典型学习曲线

```
数字猜测任务:
Episode 0, Avg Score: -2.10
Episode 10, Avg Score: 1.30
Episode 20, Avg Score: 4.50
成功率: 8/10

算术任务:
Episode 0, Avg Score: -1.20
Episode 10, Avg Score: 3.40
Episode 20, Avg Score: 7.80
正确率: 16/20
```

### 成功标准

| 任务 | 随机策略表现 | 成功标准 | 通常达成轮数 |
|------|-------------|----------|-------------|
| 数字猜测 | ~-2分 | 成功率>70% | 20-30轮 |
| 序列记忆 | ~-4分 | 平均分>3 | 25-40轮 |
| 简单算术 | ~0分 | 正确率>80% | 15-25轮 |

## 🛠️ 自定义实验

### 创建新的数字任务

```python
class YourCustomEnv(gym.Env):
    def __init__(self):
        # 定义动作和状态空间
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=1, shape=(state_dim,))
    
    def reset(self, seed=None):
        # 重置环境，返回初始状态
        return self._get_obs(), {}
    
    def step(self, action):
        # 执行动作，返回 (状态, 奖励, 完成, 截断, 信息)
        return new_state, reward, done, False, info
```

### 验证新算法

```python
# 1. 先在快速任务上验证
quick_scores = quick_train(SimpleArithmeticEnv(), episodes=30)

# 2. 如果快速任务成功，再测试经典任务
if np.mean(quick_scores[-10:]) > 5:
    classic_scores = train_reinforce('CartPole-v1', episodes=1000)
```

## 📚 扩展阅读

### 相关算法
- **REINFORCE**: 最基础的策略梯度算法
- **Actor-Critic**: 加入价值函数的改进版本
- **PPO**: 更稳定的策略优化算法
- **A2C/A3C**: 异步的Actor-Critic方法

### 更多验证环境推荐
- **Gym经典控制**: MountainCar, Acrobot, Pendulum
- **Gym离散**: FrozenLake, Taxi, Blackjack  
- **自定义网格世界**: GridWorld, CliffWalking
- **Atari游戏**: Pong, Breakout (需要更长训练时间)

## 🤝 贡献

欢迎提交新的快速验证任务！理想的任务应该：
- ⚡ 训练时间 < 1分钟
- 🎯 有明确的成功/失败标准  
- 📊 学习曲线清晰可观察
- 🔧 便于调试和理解

## 📄 许可证

MIT License - 欢迎学习和使用！ 