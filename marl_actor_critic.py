import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the Actor-Critic neural networks
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)


# PPO agent class
class PPOAgent:
    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.policy_net = PolicyNetwork(input_dim, output_dim)
        self.value_net = ValueNetwork(input_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.action_space = output_dim

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, dones):
        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def update(self, states, actions, log_probs_old, returns):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.stack(log_probs_old)
        returns = torch.FloatTensor(returns)

        values = self.value_net(states).squeeze()
        advantages = returns - values.detach()

        probs = self.policy_net(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        ratios = torch.exp(log_probs - log_probs_old)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(values, returns)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


# Agent class (unchanged)
class Agent:
    def __init__(self, x, y):
        self.position = (x, y)

    def get_state(self):
        return [self.position[0] / 9, self.position[1] / 9]  # Normalized state

    def move(self, action_index):
        action_map = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = action_map[action_index]
        new_x = max(0, min(9, self.position[0] + dx))
        new_y = max(0, min(9, self.position[1] + dy))
        self.position = (new_x, new_y)

# Initialize environment and PPO agents
GRID_SIZE = (10, 10)
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
NUM_AGENTS = 3

agents = [Agent(random.randint(0, 9), random.randint(0, 9)) for _ in range(NUM_AGENTS)]
ppo_agents = [PPOAgent(input_dim=2, output_dim=len(ACTIONS)) for _ in range(NUM_AGENTS)]

rewards_per_episode = []
avg_rewards = []

# Training loop
for episode in range(1000):
    states, actions, log_probs, rewards, dones = [], [], [], [], []
    total_reward = 0

    for agent, ppo in zip(agents, ppo_agents):
        done = False
        ep_rewards = []
        ep_states = []
        ep_actions = []
        ep_log_probs = []
        ep_dones = []

        agent.position = (random.randint(0, 9), random.randint(0, 9))  # Reset position

        for _ in range(50):  # Max steps per agent
            state = agent.get_state()
            action, log_prob = ppo.select_action(state)
            agent.move(action)
            next_state = agent.get_state()

            reward = 10 if agent.position == (9, 9) else -1
            done = agent.position == (9, 9)

            ep_states.append(state)
            ep_actions.append(action)
            ep_log_probs.append(log_prob)
            ep_rewards.append(reward)
            ep_dones.append(done)

            if done:
                break

        returns = ppo.compute_returns(ep_rewards, ep_dones)
        ppo.update(ep_states, ep_actions, ep_log_probs, returns)
        total_reward += sum(ep_rewards)

    rewards_per_episode.append(total_reward)

    if episode % 100 == 0 and episode != 0:
        avg = np.mean(rewards_per_episode[-100:])
        avg_rewards.append(avg)
        print(f"Episode {episode} | Avg Reward (last 100): {avg:.2f}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(100, 100 * len(avg_rewards) + 1, 100), avg_rewards, label="Avg Reward (per 100 eps)", color="blue")
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("PPO Learning Progress")
plt.grid(True)
plt.legend()
plt.show()
