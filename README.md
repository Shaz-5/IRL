# Inverse Reinforcement Learning Experiments

This repository implements and showcases experiments based on the paper "Algorithms for Inverse Reinforcement Learning" by Ng & Russell (2000).

## Experiment Descriptions

### 1. 5 x 5 Grid World

In the initial experiment, a 5 x 5 grid world is used. The agent starts from the lower-left grid square and navigates to the absorbing upper-right grid square. The actions correspond to the four compass directions, but with a 30% chance of moving in a random direction instead. The objective is to recover the reward structure given the policy and problem dynamics.

<p align="center">
  <img src="Results/Discrete%20Grid%20World/Agent%20Trajectory%2001.gif" width="248" />
  <img src="Results/Discrete%20Grid%20World/Agent%20Trajectory%2008.gif" width="248" />
  <img src="Results/Discrete%20Grid%20World/Agent%20Trajectory%2015.gif" width="248" />
</p>

#### Results:
- Obtained a reward function from a given policy which closely approximated the true reward.
- Also derived a reward funtion by observing the policy of a trained agent.

<p align="center">
  <img src="Results/Discrete%20Grid%20World/True%20Reward%20Function.png" width="250" heigth="250" />
  <img src="Results/Discrete%20Grid%20World/Obtained%20Reward%20Function.png" width="250" heigth="250" />
  <img src="Results/Discrete%20Grid%20World/Obtained%20Reward%20Function%20[from%20Expert].png" width="250" heigth="250" />
</p>

### 2. Mountain Car Task

The second experiment involves the "mountain-car" task, where the goal is to reach the top of the hill. The true, undiscounted, reward is -1 per step until reaching the goal. The state is the car's position and velocity, and the state space is continuous.

#### Results:
- Using a reward function based on the car's position and 26 Gaussian-shaped basis functions, the algorithm produces a reward function that captures the structure of the true reward.

<p align="center">
  <img src="Results/Mountain%20Car/Obtained%20Reward%20Function%20from%20IRL%20%5BMountain%20Car%5D.png" alt="Obtained Reward Function" width="420" height="350">
</p>

### 3. Continuous Grid World

The final experiment applies the sample-based algorithm to a continuous version of the 5 x 5 grid world. The state space is [0, 1] × [0, 1], and actions move the agent 0.2 in the intended direction with added noise. The true reward is 1 in a non-absorbing square [0.8, 1] × [0.8, 1], and 0 everywhere else.

#### Results:
- The algorithm, using linear combinations of two-dimensional Gaussian basis functions and produces reasonable solutions.

<p align="center">
  <img src="Results/Continuous%20Grid%20World/Actual%20Reward%20Function%20%5BContinous%20Grid%20World%5D.png" width="400" />
  <img src="Results/Continuous%20Grid%20World/Obtained%20Reward%20Function%20%5BContinous%20Grid%20World%5D.png" width="400" />
</p>


## Documentation

Feel free to explore my introductory presentation to Inverse Reinforcement Learning (IRL) and also get an overview of the experiments conducted.

- [Introduction to IRL and Experiments](Docs/Inverse%20Reinforcement%20Learning%20Introduction.pdf)


## References

- Ng, A., & Russell, S. (2000). Algorithms for Inverse Reinforcement Learning.
- ShivinDass. (n.d.). GitHub - ShivinDass/inverse_rl: GitHub: github.com/ShivinDass/inverse_rl
