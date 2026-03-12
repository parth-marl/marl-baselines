# Treasure Island — Q-Learning Experiment

Small reinforcement learning experiment I built while learning **tabular Q-learning**.

The goal was not to build something fancy, but to **understand how the learning loop actually works** and how values propagate through a state space.

Most of the time spent on this project was debugging the environment, not the algorithm — which turned out to be the real lesson.

---

## Environment

A simple world with **120 states** arranged in a linear index.

The agent starts at **state 0** and tries to reach the **treasure at state 119**.

There are **50 traps randomly placed** across the map.

If the agent lands on a trap the episode ends.

Actions available to the agent:

```
←  move left  (state -1)
→  move right (state +1)
↑  move up    (state -5)
↓  move down  (state +5)
J  jump       (state +10)
```

The agent does **not know the map** beforehand.
It learns purely from rewards.

---

## Rewards

```
+10  reaching the treasure
-5   falling into a trap
-0.1 step penalty
```

The step penalty encourages the agent to find **shorter paths instead of wandering**.

---

## Learning Algorithm

Classic **Q-learning** with epsilon-greedy exploration.

Update rule:

```
Q(s,a) = Q(s,a) + α (r + γ max(Q(s')) − Q(s,a))
```

Hyperparameters used:

```
learning_rate (α)  = 0.7
discount_factor (γ) = 0.9
epsilon_start       = 0.3
episodes            = 100000
```

Epsilon decays during training so the agent gradually shifts from **exploration → exploitation**.

---

## What the Agent Learns

Over many episodes the agent builds a **Q-table (120 × 5)** where each entry represents the value of taking an action in a state.

Eventually the agent discovers safe trajectories that avoid traps and reach the treasure.

The learned policy is simply:

```
choose action with highest Q value
```

---

## Diagnostics / Visualizations

Two graphs are generated after training.

### 1. Max Q-Value per State

Shows how valuable each state is according to the learned policy.

States that lie on successful paths toward the goal accumulate higher values as the reward propagates backward through the state space.

---

### 2. State Visit Frequency

Tracks how often each state was visited during training.

This reveals the **actual routes the agent prefers** and which parts of the map it avoids.

In practice the agent converges to a small set of high-traffic “corridor” states that form an efficient path to the treasure.

---

## What This Project Taught Me

A few things became very obvious while building this:

• Most RL bugs come from **environment design**, not the learning algorithm
• Reward shaping drastically changes agent behavior
• Graphs can be misleading without proper diagnostics
• Value propagation is the core mechanism behind reinforcement learning

---

## Run the Project

```
python treasure_island_q_learning.py
```

Dependencies:

```
numpy
matplotlib
random
```

---

## Future Experiments

Possible extensions I want to try:

• Convert this into a **true gridworld** instead of a 1D state index
• Visualize the **policy as arrows on the map**
• Implement **Deep Q-Networks** instead of tabular Q-learning
• Introduce stochastic traps or moving hazards

---

This project is mainly a **learning sandbox** for understanding reinforcement learning fundamentals.
