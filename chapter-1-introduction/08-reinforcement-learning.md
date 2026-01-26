# Reinforcement Learning

## What is Reinforcement Learning?

**Reinforcement Learning (RL)**: Learning by trial and error through interactions with an environment, guided by rewards.

```
Agent learns optimal behavior by:
✓ Taking actions
✓ Observing outcomes
✓ Receiving rewards/penalties
✓ Adjusting strategy
```

## Core Components

```
┌─────────────────────────────────────────┐
│    REINFORCEMENT LEARNING LOOP          │
├─────────────────────────────────────────┤
│                                         │
│     ┌─────────┐                         │
│     │  Agent  │                         │
│     └────┬────┘                         │
│          │                              │
│   Action │   ↓                          │
│     ┌────▼────────┐                     │
│     │ Environment │                     │
│     └────┬────────┘                     │
│          │                              │
│  State + │   ↓                          │
│  Reward  │                              │
│          ↓                              │
│     ┌─────────┐                         │
│     │  Agent  │ (learns policy)         │
│     └─────────┘                         │
│                                         │
└─────────────────────────────────────────┘
```

### Key Elements

1. **Agent**: The learner/decision maker
2. **Environment**: The world the agent interacts with
3. **State (s)**: Current situation
4. **Action (a)**: What agent can do
5. **Reward (r)**: Feedback signal (+/-)
6. **Policy (π)**: Strategy for choosing actions

## The RL Process

```
STEP 1: Agent observes state
        State: "Player at position (3, 5)"

STEP 2: Agent chooses action (based on policy)
        Action: "Move right"

STEP 3: Environment responds
        New state: "Player at position (4, 5)"
        Reward: +1 (moved closer to goal)

STEP 4: Agent updates policy
        "Moving right in this state was good!"

REPEAT: Learn optimal policy through experience
```

## Simple Example: Grid World

```
Grid World Game:
┌───┬───┬───┬───┐
│ S │   │   │   │  S = Start
├───┼───┼───┼───┤
│   │ X │   │   │  X = Obstacle (-10)
├───┼───┼───┼───┤
│   │   │   │   │  G = Goal (+100)
├───┼───┼───┼───┤
│   │   │   │ G │  Empty = (-1)
└───┴───┴───┴───┘

Actions: Up, Down, Left, Right
Goal: Learn to reach G while avoiding X
```

```python
import numpy as np

class GridWorld:
    def __init__(self):
        self.grid_size = 4
        self.state = (0, 0)  # Start position
        self.goal = (3, 3)
        self.obstacle = (1, 1)
    
    def step(self, action):
        # Action: 0=up, 1=down, 2=left, 3=right
        x, y = self.state
        
        if action == 0: x = max(0, x-1)
        elif action == 1: x = min(3, x+1)
        elif action == 2: y = max(0, y-1)
        elif action == 3: y = min(3, y+1)
        
        self.state = (x, y)
        
        # Rewards
        if self.state == self.goal:
            reward = 100
            done = True
        elif self.state == self.obstacle:
            reward = -10
            done = False
        else:
            reward = -1  # Small penalty for each step
            done = False
        
        return self.state, reward, done
    
    def reset(self):
        self.state = (0, 0)
        return self.state

# Usage
env = GridWorld()
state = env.reset()
state, reward, done = env.step(3)  # Move right
print(f"State: {state}, Reward: {reward}")
```

## Q-Learning: Classic RL Algorithm

**Q-value**: Expected total reward for taking action `a` in state `s`.

```
Q(s, a) = Expected future rewards
```

### Q-Table

```
State-Action table:

         Up    Down   Left   Right
(0,0)  [ 5.2   3.1   -2.0   8.4  ]  ← Best action: Right (8.4)
(0,1)  [ 6.3   4.2    7.1   2.3  ]  ← Best action: Left (7.1)
(1,0)  [ 3.2   1.8    4.5   9.2  ]  ← Best action: Right (9.2)
...

Agent learns these values through experience!
```

### Q-Learning Algorithm

```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions):
        # Initialize Q-table with zeros
        self.Q = np.zeros((n_states, n_actions))
        
        # Hyperparameters
        self.alpha = 0.1   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.1 # Exploration rate
    
    def choose_action(self, state):
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            # Explore: Random action
            return np.random.randint(0, self.Q.shape[1])
        else:
            # Exploit: Best known action
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.Q[next_state])
        
        # TD (Temporal Difference) update
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        
        self.Q[state, action] += self.alpha * td_error

# Training
agent = QLearning(n_states=16, n_actions=4)
env = GridWorld()

for episode in range(1000):
    state = env.reset()
    total_reward = 0
    
    for step in range(100):
        # Choose action
        action = agent.choose_action(state)
        
        # Take action
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Learn from experience
        agent.update(state, action, reward, next_state)
        
        state = next_state
        
        if done:
            break
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# After training, Q-table contains optimal policy!
```

## Key Formulas

### Q-Learning Update

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                       ↑       ↑
                   reward   future value
```

Where:
- `α` (alpha) = Learning rate (0 to 1)
- `γ` (gamma) = Discount factor (0 to 1)
- `r` = Immediate reward
- `s'` = Next state
- `max Q(s',a')` = Best future Q-value

### Policy

```
π(s) = argmax Q(s,a)
         a
```
(Choose action with highest Q-value)

## Exploration vs Exploitation

**The RL Dilemma:**

```
EXPLOITATION:              EXPLORATION:
Use best known action      Try new actions
Safe, predictable          Risky, uncertain
May miss better options    May find better options

┌──────────────┐          ┌──────────────┐
│  Known path  │          │  Unknown     │
│  Reward: 10  │          │  Reward: ??  │
└──────────────┘          └──────────────┘
     ↑                         ↑
  Exploit                   Explore
```

**Epsilon-Greedy Strategy:**
```python
if random() < epsilon:
    action = random_action()  # Explore (10%)
else:
    action = best_action()    # Exploit (90%)
```

## Deep Q-Network (DQN)

For complex environments, use neural networks instead of tables.

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # Q-value for each action
        )
    
    def forward(self, state):
        return self.network(state)

# Usage
dqn = DQN(state_dim=4, action_dim=2)
state = torch.tensor([0.1, 0.5, -0.3, 0.8])
q_values = dqn(state)
# Output: [2.3, 4.1] → Choose action 1 (higher Q-value)
action = torch.argmax(q_values)
```

## Real-World Example: Game Playing

```python
# Simplified Atari game RL

class AtariAgent:
    def __init__(self):
        self.dqn = DQN(state_dim=84*84*4, action_dim=6)
        self.optimizer = torch.optim.Adam(self.dqn.parameters())
        self.memory = []  # Experience replay buffer
    
    def train_step(self):
        # Sample random batch from memory
        batch = random.sample(self.memory, 32)
        
        for state, action, reward, next_state, done in batch:
            # Current Q-value
            q_value = self.dqn(state)[action]
            
            # Target Q-value
            if done:
                target = reward
            else:
                target = reward + 0.99 * torch.max(self.dqn(next_state))
            
            # Loss
            loss = (q_value - target) ** 2
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Agent learns to play game through trial and error!
```

## What Humans Define vs What AI Learns

```
┌─────────────────────────────────────────┐
│  HUMANS DEFINE:                         │
├─────────────────────────────────────────┤
│  ✓ State representation                 │
│  ✓ Available actions                    │
│  ✓ Reward function                      │
│  ✓ Environment rules                    │
│  ✓ Training algorithm                   │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  AI LEARNS:                             │
├─────────────────────────────────────────┤
│  ✓ Which actions to take (policy)       │
│  ✓ When to take them                    │
│  ✓ Long-term strategy                   │
│  ✓ How to maximize rewards              │
└─────────────────────────────────────────┘
```

## Types of RL

### 1. Value-Based (Q-Learning, DQN)
Learn Q-values, derive policy from them.

### 2. Policy-Based (Policy Gradient)
Learn policy directly.

```python
# Policy network outputs action probabilities
class PolicyNetwork(nn.Module):
    def forward(self, state):
        logits = self.network(state)
        probs = torch.softmax(logits, dim=-1)
        return probs  # [0.1, 0.7, 0.2] → Likely choose action 1
```

### 3. Actor-Critic
Combines value-based and policy-based.

```
Actor: Learns policy (what to do)
Critic: Learns value function (how good is state)
```

## Discount Factor (γ)

Controls how much to value future rewards.

```
γ = 0:   Only care about immediate reward
γ = 0.5: Balance immediate and future
γ = 0.99: Heavily value future rewards

Example:
Reward sequence: [1, 1, 1, 1, 100]

γ = 0:   Value = 1 (only first)
γ = 0.9: Value = 1 + 0.9 + 0.81 + 0.73 + 65.61 = 69.05
γ = 1:   Value = 104 (all rewards equally)
```

## Reward Shaping

Designing rewards to guide learning.

```python
# Sparse rewards (hard to learn):
def reward_sparse(state):
    if state == goal:
        return 100
    else:
        return 0  # No feedback!

# Shaped rewards (easier to learn):
def reward_shaped(state):
    if state == goal:
        return 100
    else:
        # Distance-based reward
        distance = manhattan_distance(state, goal)
        return -distance  # Guides toward goal
```

## Applications

| Domain | Agent | Actions | Rewards |
|--------|-------|---------|---------|
| Games | Player | Moves | Win/lose points |
| Robotics | Robot | Motor controls | Task completion |
| Trading | Trader | Buy/sell/hold | Profit/loss |
| Chatbot | Bot | Response choice | User satisfaction |
| Autonomous driving | Car | Steer/brake/accelerate | Safe arrival |

## Analogy: Learning to Ride a Bike

```
Traditional ML:
"Here are 1000 labeled examples of good bike riding"
→ Learn from expert demonstrations

Reinforcement Learning:
"Try riding. You'll fall (negative reward).
 Keep trying. You'll balance (positive reward).
 Eventually, you'll learn!"
→ Learn from experience

Like a child learning:
- No explicit instructions
- Trial and error
- Reward: staying balanced
- Penalty: falling
- Gradually improves through practice
```

## Training Visualization

```
Training Progress:

Episode 0:    Random actions, poor performance
●─────────────────────────── (crashed)
Reward: -50

Episode 100:  Starting to learn
●──────●──────●─────────●─── (improving)
Reward: -10

Episode 500:  Good strategy emerging
●─────●─────●─────●─────G   (reached goal sometimes)
Reward: +50

Episode 1000: Optimal policy learned
●───●───●───●───G            (efficient path)
Reward: +95
```

## Challenges

### 1. Credit Assignment
Which action led to reward?

```
Action sequence: [A, B, C, D] → Reward: +100
Which action(s) were important?
```

### 2. Sparse Rewards
Rewards are rare, hard to learn.

**Solution:** Reward shaping, curriculum learning

### 3. Sample Efficiency
RL often needs many trials.

**Solution:** Experience replay, transfer learning

## Key Takeaway

```
Reinforcement Learning = Learning by Doing

🎯 Core Idea
  - Agent interacts with environment
  - Receives rewards/penalties
  - Learns optimal behavior

🔄 The Loop
  1. Observe state
  2. Choose action (policy)
  3. Receive reward
  4. Update knowledge
  5. Repeat

🧠 What's Learned
  - NOT given labeled data
  - NOT told what to do
  - LEARNS through experience
  - DISCOVERS optimal strategy

📊 Methods
  - Q-Learning: Learn action values
  - Policy Gradient: Learn policy directly
  - Actor-Critic: Learn both

🎮 Applications
  - Game playing (AlphaGo, Atari)
  - Robotics (manipulation, walking)
  - Autonomous systems (driving, drones)
  - Optimization (resource allocation)

Success = Find actions that maximize long-term rewards!
```
