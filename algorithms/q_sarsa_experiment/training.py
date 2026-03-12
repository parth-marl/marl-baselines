import numpy as np
import random
from environment import GridWorld

def state_to_index(state, size):
    return state[0] * size + state[1]

def choose_action(Q, state_index, epsilon):
    
    if np.random.rand() < epsilon:
        return np.random.randint(4)  # explore
    
    return np.argmax(Q[state_index])  # exploit

def train_q_sarsa(env, episodes=5000, alpha=0.1, gamma=0.95, epsilon=0.2):

    state_space = env.size * env.size
    action_space = 4

    Q = np.zeros((state_space, action_space))

    for episode in range(episodes):

        state = env.reset()
        state_index = state_to_index(state, env.size)

        action = choose_action(Q, state_index, epsilon)

        done = False
        steps = 0

        while not done and steps < 200:

            next_state, reward, done = env.step(action)
            next_index = state_to_index(next_state, env.size)

            next_action = choose_action(Q, next_index, epsilon)

            # ---- Q-SARSA Hybrid Weight ----
            n = max(0, 1 - episode / episodes)

            # SARSA future
            q_sarsa = Q[next_index, next_action]

            # Q-learning future
            q_ql = np.max(Q[next_index])

            # Hybrid future
            future = n * q_sarsa + (1 - n) * q_ql

            # Target
            target = reward + gamma * future

            # Update
            Q[state_index, action] += alpha * (target - Q[state_index, action])

            # Move forward
            state = next_state
            state_index = next_index
            action = next_action

            steps += 1

    return Q

def print_policy(Q, env):

    arrows = ["←","→","↓","↑"]

    for r in range(env.size):
        row = ""
        for c in range(env.size):

            if (r,c) == env.goal:
                row += " G "
                continue

            if [r,c] in env.traps:
                row += " X "
                continue

            idx = state_to_index((r,c), env.size)
            best_action = np.argmax(Q[idx])

            row += " " + arrows[best_action] + " "

        print(row)

env = GridWorld()

Q = train_q_sarsa(env)

print("Q-table learned:\n")
print(Q)
print("\nLearned Policy:\n")
print_policy(Q, env)
