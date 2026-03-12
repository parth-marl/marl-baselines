# Importing Liabraries
import numpy as np
from matplotlib import pyplot as plt
import random
# Defining Environment
n_states = 120
n_action = 5
Q = np.zeros((n_states, n_action))
# print(Q)
learning_factor = 0.7
discount_factor = 0.9
epsilon = 0.3

SEED = 50
np.random.seed(SEED)
random.seed(SEED)

goal = 119
possible_traps = [s for s in range(1, n_states-1) if s != goal]
traps = np.random.choice(possible_traps, size=50, replace=False)

episodes = 100000
visit_counts = np.zeros(n_states)
# function to move onto next state
def get_next_state(state, action):

    if action == 0:#left
        return max(0, state - 1)
    elif action == 1:#right
        return min(n_states - 1, state + 1)
    elif action == 2: #up
        return max(0, state - 5)
    elif action ==3:
        return min(n_states - 1, state + 5)
    elif action == 4:
        return min(n_states - 1, state + 10)
    else:
        return state
# Core learning loop
for ep in range(episodes):
    state = 0
    steps = 0
    epsilon = max(0.01, epsilon * 0.9995)

    while True:
        random_int = np.random.rand()
        # Exploration 
        if random_int < epsilon:
            action = np.random.randint(n_action)
        # Exploitation 
        else:
            action = np.argmax(Q[state])

        next_state = get_next_state(state, action)

        if next_state == goal:
            reward = 10
            done = True
        elif next_state in traps:
            reward = -5
            done = True
        else:
            reward = -0.1
            done = False

        Q[state, action] += learning_factor * (reward + discount_factor*np.max(Q[next_state]) - Q[state, action])

        visit_counts[state] += 1
        
        if done:
            break
        state = next_state
    steps += 1
    if steps > 200:
        break
# Plot max Q-value per state
plt.figure(figsize=(12,5))
plt.plot(np.max(Q, axis=1))
plt.title("Treasure Island (120 states) - Max Q-value per State")
plt.xlabel("State Index")
plt.ylabel("Max Q-value")
plt.grid(True)
plt.show()
#print learned policy for first 50 states
actions_map = ['←','→','↑','↓','J']
policy = [actions_map[np.argmax(Q[s])] for s in range(50)]
print("Learned policy (first 50 states):")
print(policy)
plt.figure(figsize=(12,5))
plt.plot(visit_counts)
plt.title("State Visit Frequency")
plt.xlabel("State Index")
plt.ylabel("Visits")
plt.grid(True)
plt.show()