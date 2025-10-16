# Symbiotic-Design-for-Multi-Robot-Systems
This repository contains the implementation for my Master’s thesis, “Symbiotic Design in Multi-Robot Ecosystems.” It explores cooperation among heterogeneous robots using MARL with Actor-Critic and PPO methods, applying mutualistic behavior, A* pathfinding, and collision avoidance to improve task efficiency in a grid environment.

---

# Overview

The goal of this project is to explore symbiotic task allocation between two types of delivery robots — small and large — operating in a shared grid environment.
By applying Actor-Critic and Proximal Policy Optimization (PPO) methods, the study investigates how mutualism-inspired cooperation can improve efficiency, reduce collisions, and optimize delivery times.

---

# Key features
Key Features

- Grid-based simulation environment (10x10)
- Two heterogeneous robot types (small and large)
- A* pathfinding and collision avoidance
- Actor-Critic and PPO reinforcement learning
- Mutualism-based coordination and task-sharing
- Visualization of robot movements and performance metrics

--- 

# Repository Structure
```bash
├── Agent.py
├── README.md
├── delivery.py
├── marl_actor_critic.py
├── packageInfo.py
├── time.py
```

---

# How to use
Prerequisites
- Python 3.8+
- numpy
- matplotlib
- pandas
- heapq

---

# How to run
Run the simulation:

```bash
python delivery.py
```

---

# Output examples:
Example Outputs:
- Visualization of robots navigating a 10x10 grid, avoiding collisions, and completing delivery tasks.
- Learning curves showing reward improvement and reduction in collisions over training episodes.

<img width="715" height="699" alt="Screenshot 2025-06-03 155738 (1)" src="https://github.com/user-attachments/assets/fbce43b9-75c6-495f-bcd3-8d557ba57541" />
<p align="center">
  <img src="Screenshot 2025-06-03 155738 (1).png" width="400"><br>
  <em>Figure 1: Grid environment with small and big robots, pickup/drop-off points, and shelf obstacles (yellow).</em>
</p>
<img width="627" height="154" alt="Screenshot 2025-06-04 115601 (1)" src="https://github.com/user-attachments/assets/1ae17f43-d818-4a86-ab5c-7b58350d55f0" />
<p align="center">
  <img src="Screenshot 2025-06-04 115601 (1).png" width="400"><br>
  <em>Figure 2: Completion Time for all the robots</em>
</p>
<img width="1235" height="617" alt="Screenshot 2025-04-25 120935 (1)" src="https://github.com/user-attachments/assets/92ff0818-bbc6-4553-bafe-9579751fdc1f" />
<p align="center">
  <img src="Screenshot 2025-04-25 120935 (1).png" width="400"><br>
  <em>Figure 3: Learning curve</em>
</p>

---
# Contact
For questions, reach out via [LinkedIn](https://www.linkedin.com/in/vinay-mehra-5842b0106/) or mvinay908@gmail.com.

---

# License
This project is for academic and research purposes. For licensing inquiries, please contact the author.
