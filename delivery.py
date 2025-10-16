import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import torch
from Agent import Agent
from marl_actor_critic import PPOAgent  # Assuming you have PPOAgent class
from packageInfo import generate_random_pickup, avoid_collision, a_star_search, calculate_total_time

# Grid setup
GRID_SIZE = (10, 10)
SMALL_CAR_STARTS = [(0, 0), (1, 0), (0, 1)]
BIG_CAR_STARTS = [(9, 9), (8, 9), (9, 8)]
DELIVERY_END_SMALL = (9, 0)
DELIVERY_END_BIG = (0, 9)
SHELF_LOCATIONS = [(3, 6), (4, 6), (5, 6), (6, 6), (3, 3), (4, 3), (5, 3), (6, 3)]
INTERMEDIATE_POINT = (5, 5)  # Relay point for mutualism

# Road costs
hroads = np.ones(GRID_SIZE)
vroads = np.ones(GRID_SIZE)
hroads[3, :] = 2
vroads[:, 6] = 3

# Mutualism task: Big picks up and drops at intermediate, Small continues
mutual_pickup = generate_random_pickup(GRID_SIZE, BIG_CAR_STARTS[0], INTERMEDIATE_POINT, SHELF_LOCATIONS)
path_big_pickup = a_star_search(BIG_CAR_STARTS[0], mutual_pickup, hroads, vroads, SHELF_LOCATIONS)
path_big_drop = a_star_search(mutual_pickup, INTERMEDIATE_POINT, hroads, vroads, SHELF_LOCATIONS)
mutual_path_big = path_big_pickup + path_big_drop[1:]

path_small_pick = a_star_search(INTERMEDIATE_POINT, DELIVERY_END_SMALL, hroads, vroads, SHELF_LOCATIONS)
path_small_return = a_star_search(DELIVERY_END_SMALL, SMALL_CAR_STARTS[0], hroads, vroads, SHELF_LOCATIONS)
mutual_path_small = path_small_pick + path_small_return[1:]

# Other car paths
pickups_small = [generate_random_pickup(GRID_SIZE, SMALL_CAR_STARTS[i], DELIVERY_END_SMALL, SHELF_LOCATIONS)
                 for i in range(1, 2)]
pickups_big = [generate_random_pickup(GRID_SIZE, BIG_CAR_STARTS[i], DELIVERY_END_BIG, SHELF_LOCATIONS,
                                      existing_pickups=pickups_small)
               for i in range(1, 3)]

paths_small = [mutual_path_small]
paths_big = [mutual_path_big]

for i in range(1, 3):
    if i < len(pickups_small):
        pickup = pickups_small[i-1]
        path_pickup = a_star_search(SMALL_CAR_STARTS[i], pickup, hroads, vroads, SHELF_LOCATIONS)
        path_delivery = a_star_search(pickup, DELIVERY_END_SMALL, hroads, vroads, SHELF_LOCATIONS)
        path_return = a_star_search(DELIVERY_END_SMALL, SMALL_CAR_STARTS[i], hroads, vroads, SHELF_LOCATIONS)
        paths_small.append(path_pickup + path_delivery[1:] + path_return[1:])
    else:
        paths_small.append([SMALL_CAR_STARTS[i]] * 5)

    pickup = pickups_big[i-1]
    path_pickup = a_star_search(BIG_CAR_STARTS[i], pickup, hroads, vroads, SHELF_LOCATIONS)
    path_delivery = a_star_search(pickup, DELIVERY_END_BIG, hroads, vroads, SHELF_LOCATIONS)
    path_return = a_star_search(DELIVERY_END_BIG, BIG_CAR_STARTS[i], hroads, vroads, SHELF_LOCATIONS)
    paths_big.append(path_pickup + path_delivery[1:] + path_return[1:])

all_paths = avoid_collision(*(paths_small + paths_big))
total_time = calculate_total_time(all_paths)
print(f"Total time for all cars to deliver their packages: {total_time} units of time")

# PPO agent
NUM_AGENTS = 6
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
ppo_agent = PPOAgent(input_dim=2, output_dim=len(actions), lr=0.01, gamma=0.99, eps_clip=0.1)
agents = [Agent(random.randint(0, GRID_SIZE[0]-1), random.randint(0, GRID_SIZE[1]-1)) for _ in range(NUM_AGENTS)]

rewards_per_episode = []
avg_rewards = []

for episode in range(1000):
    episode_reward = 0
    for agent in agents:
        state = agent.get_state()
        action, log_prob = ppo_agent.select_action(state)
        agent.move(action)
        next_state = agent.get_state()

        if next_state == INTERMEDIATE_POINT:
            reward = 5  # mutualism cooperation reward
        elif next_state == DELIVERY_END_SMALL or next_state == DELIVERY_END_BIG:
            reward = 10  # successful delivery
        else:
            reward = -1  # step penalty

        ppo_agent.update([state], [action], [log_prob], [reward])
        episode_reward += reward

    rewards_per_episode.append(episode_reward)
    if len(rewards_per_episode) >= 100:
        avg_reward = np.mean(rewards_per_episode[-100:])
        avg_rewards.append(avg_reward)
        print(f"Episode {episode} | Avg Reward (last 100 eps): {avg_reward:.2f}")

overall_avg_reward = np.mean(rewards_per_episode)
print(f"Overall Average Reward: {overall_avg_reward:.2f}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(100, 100 * len(avg_rewards) + 1, 100), avg_rewards, label="Avg Reward (per 100 eps)", color="green")
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Learning Progress - Avg Reward Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Visualization
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks(np.arange(GRID_SIZE[0] + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(GRID_SIZE[1] + 1) - 0.5, minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
ax.set_xlim(-0.5, GRID_SIZE[0] - 0.5)
ax.set_ylim(-0.5, GRID_SIZE[1] - 0.5)

# Plot base grid
def plot_grid():
    for x in range(GRID_SIZE[0]):
        for y in range(GRID_SIZE[1]):
            color = "white"
            if (x, y) in SMALL_CAR_STARTS:
                color = "yellow"
            elif (x, y) in BIG_CAR_STARTS:
                color = "black"
            elif (x, y) in [DELIVERY_END_SMALL, DELIVERY_END_BIG]:
                color = "green"
            elif (x, y) in SHELF_LOCATIONS:
                color = "yellow"
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color, alpha=0.5))

plot_grid()
ax.plot(INTERMEDIATE_POINT[0], INTERMEDIATE_POINT[1], "ro", markersize=12, label="Relay Point")
ax.legend()

# Cars
small_cars = [ax.plot([], [], "yo", markersize=10)[0] for _ in range(3)]
big_cars = [ax.plot([], [], "ko", markersize=10)[0] for _ in range(3)]

max_frames = max(len(path) for path in all_paths)

# Animation
def update(frame):
    for i in range(3):
        if frame < len(paths_small[i]):
            small_cars[i].set_data(*paths_small[i][frame])
        if frame < len(paths_big[i]):
            big_cars[i].set_data(*paths_big[i][frame])
    return small_cars + big_cars

ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=500, blit=False)
plt.show()
