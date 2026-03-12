Hybrid Q-SARSA Experiment
Overview

This project started as a small experiment while I was learning reinforcement learning.

I had already been working with Q-learning and SARSA, and I started wondering what would happen if I combined them into a single update rule.

The idea was simple:

SARSA is cautious because it learns from the action the agent actually takes.

Q-learning is optimistic because it assumes the best possible action will be taken in the future.

So I tried blending the two.

The hybrid update I experimented with looks like this:

future = n * Q(s', a') + (1 - n) * max_a' Q(s', a')
target = r + γ * future
Q(s,a) ← Q(s,a) + α (target − Q(s,a))

Where:

Q(s', a') → SARSA estimate (what the agent actually did)

max_a' Q(s', a') → Q-learning estimate (best possible future action)

n → a weight that decays over time

This means:

Early training → agent behaves more like SARSA (cautious)

Later training → agent behaves more like Q-learning (optimistic)

Why I Built This

Initially this was just curiosity.

While learning RL, I wanted to see how small changes in the update rule could affect the way an agent behaves.

At first the thought process was basically:

Q-learning is optimistic
SARSA is cautious
What happens if the agent starts cautious and gradually becomes optimistic?

So I implemented this hybrid and ran it inside a small GridWorld environment.

Mid-Journey Realization

Halfway through experimenting I realized something funny:

Ideas similar to this already exist.

Algorithms like Expected SARSA or other variants already blend policy expectations with optimal estimates.

But I kept the experiment anyway because implementing things yourself teaches way more than just reading about them.

Environment

The agent is trained in a simple GridWorld:

Start position somewhere in the grid

Traps (X) that give negative reward

A goal (G) that gives positive reward

The agent learns the optimal path over many episodes

Actions:

0 = Left
1 = Right
2 = Down
3 = Up

The Q-table stores expected returns for every state-action pair.

Results

After training, the learned Q-table looked like this:

[[-0.13958259  0.80540477  0.93866768 -0.1430065 ]
 [-0.87218543  0.50757496  1.96504174  0.18563213]
 [-1.86138399 -2.06319974  2.32652174 -1.62208815]
 [-1.63823847  0.7876967  -0.94763174 -1.5026228 ]
 [-1.24489417 -1.19517988  4.15357109 -1.0709901 ]
 [ 0.90116843  1.96293059  2.04339911 -0.1795003 ]
 [ 0.52855544  2.50244441  3.17421982  0.5366096 ]
 [-0.04328301  4.02019773  1.96088015 -0.80662439]
 [ 0.92149775  5.53562024  3.39794525 -1.23490605]
 [ 1.95155016  3.19396085  7.02305491  0.60570874]
 [ 2.00644246  3.15127291  3.20592617  0.90200206]
 [ 1.82138432  4.09214106  4.42135999  1.83391725]
 [ 2.58182851  4.3219165   5.68526014  2.14303261]
 [ 0.54920496  7.02357309  3.1893054   2.12313341]
 [ 3.8609692   5.76573418  8.49568811  4.3768361 ]
 [ 3.16425029  4.42949122  4.28115919  1.96768664]
 [ 3.14092252  5.7172896   5.66785408  3.13631071]
 [ 4.36647733  7.07221677  7.03767798  4.35044315]
 [ 5.66389227  8.49889813  8.46929676  5.58916347]
 [ 7.02349132  8.48180916 10.          7.04525993]
 [ 1.49516586  5.65372559  2.21444737  0.16373735]
 [ 4.06982716  7.06213425  5.59804989  4.23898077]
 [ 5.55809564  8.49760067  6.95113539  5.55933311]
 [ 6.91617345 10.          8.48170579  6.97777012]
 [ 0.          0.          0.          0.        ]]

From this Q-table we can extract the best policy the agent learned:

 ↓  ↓  ↓  →  ↓
 ↓  X  X  X  ↓
 ↓  ↓  ↓  X  ↓
 →  X  →  →  ↓
 →  →  →  →  G

Where:

→ ← ↑ ↓ represent the best action in that state

X are traps

G is the goal

The agent successfully learned to navigate around traps and reach the goal.

What I Learned

This small experiment helped me understand a few things:

Small modifications in the Bellman update can change agent behavior significantly.

Blending optimistic and cautious updates can affect risk-taking during learning.

Actually implementing algorithms gives much deeper intuition than just reading about them.

Even if the idea isn't completely new, building it from scratch helped me understand how value propagation works in RL.

Final Thoughts

This project is mostly an educational experiment while learning reinforcement learning.

The goal wasn’t to invent a brand-new algorithm, it was to explore how these ideas behave when you start tweaking them.

And honestly, watching the agent slowly figure out the grid was pretty fun.