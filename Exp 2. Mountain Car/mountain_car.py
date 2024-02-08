import gym
import numpy as np
import random, copy
import pickle

# function to take a continuous state 's' from the environment and discretize it into a grid position
def get_state(s):
    xpos = min(discretization - 1, round((s[0] - xmin) / x_binsize))    # s[0] and s[1] - position and velocity of the agent
    vpos = min(discretization - 1, round((s[1] - vmin) / v_binsize))
    return int(xpos), int(vpos)     # indices in the discretized state space

# create the Mountain Car environment
env = gym.make('MountainCar-v0')
# env.seed(42)

# hyperparameters
alpha = 0.1     # learning rate
gamma = 0.99    # discount rate
epsilon = 0.05  # exploration rate

# initialize Q-table with zeros
Q = [[[0 for _ in range(3)] for _ in range(120)] for _ in range(120)]
Q_optim = copy.deepcopy(Q)    # copy of Q to store the optimized Q-values later

# Minimum steps needed (initialized to maximum possible steps)
min_steps = env._max_episode_steps + 1

# environment information
e = env.env
discretization = 120    # number of bins

# discretize state space
xmin = e.min_position                         # min value for position
xmax = e.max_position                         # max value for position
x_binsize = (xmax - xmin) / discretization    # size of each bin in the discretized space

vmin = -1 * e.max_speed                       # min value for velocity
vmax = e.max_speed                            # max value for velocity
v_binsize = (vmax - vmin) / discretization


# print("Maximum Episode Steps:", env._max_episode_steps)
# print("\nDiscretization Details:\n")
# print(f"Position: {xmin} to {xmax}, Bin Size: {x_binsize}")
# print(f"Velocity: {vmin} to {vmax}, Bin Size: {v_binsize}")

# time = 0

# while time < 20000:   # or min_steps > 260:
#     obs = env.reset()
#     score = 0
#     steps = 0

#     greed_pol = random.randint(0, 9)

#     while True:     # episode
#         steps += 1
#         # if time % 2000 == 0:
#         #     env.render()

#         x, v = get_state(obs)
#         a = random.randint(0, 2)

#         # xxploration or exploitation based on epsilon-greedy strategy
#         if greed_pol == 0 or random.random() < 1 - epsilon:
#             a = np.argmax(np.array(Q[x][v]))

#         obs, R, done, info = env.step(a)

#         x1, v1 = get_state(obs)

#         score += R

#         Q[x][v][a] += alpha * (R + gamma * max(Q[x1][v1]) - Q[x][v][a])   # Q(s, a) ← Q(s, a) + α ( R + γ * max{a'} * Q(s', a') - Q(s, a)) Bellman Eq

#         if done:    # episode terminated
#             break

#     time += 1
#     if time%2000 == 0:
#         print(f"Episode: {time} -> Steps: {steps}")

#     # update the lowest steps and Q values
#     if greed_pol == 0 and steps < min_steps:
#         min_steps = steps
#         Q_optim = copy.deepcopy(Q)

# print(f"Min Steps: {min_steps}")

# # Saving Q_optim

# file_path = 'Q_opt.pkl'
# with open(file_path, 'wb') as file:
#     pickle.dump(Q, file)

# # evaluate the trained Q-learning agent's performance in the environment

# while input('Continue?: ').lower() == "y":

#     obs = env.reset(-0.5)

#     while True:
#         env.render()
#         x, v = get_state(obs)
#         a = np.argmax(np.array(Q[x][v]))
#         obs, R, done, info = env.step(a)

#         if done:
#             break

# function for calculating the probability density function (PDF) of a normal distribution

from math import exp, sqrt, pi

def calc_pdf(a, m, s):
    a = (a - m) / s
    return (exp(-(a) ** 2 / (2)) / (sqrt(2 * pi))) / s

def get_value_function(Q,mean,scale,i,alpha=0.1,gamma=0.99,epsilon=0.05):

    V=[[0 for v in range(120)] for x in range(120)]

    time=0
    while time<10000:     # or min_steps>260:
        obs = env.reset()

        while True:
            x,v = get_state(obs)
            #a=random.randint(0,2)
            #if random.random()<1-epsilon:
              #print(Q[x][v])
            a=np.argmax(np.array(Q[x][v]))      # choose an action using the policy represented by Q-values

            obs,R,done,info=env.step(a)
            R = calc_pdf(x,mean,scale)           # calculate a reward using a probability density function (PDF)

            x1,v1 = get_state(obs)			          # get next state (x1, v1)

            V[x][v] += alpha*(R + gamma*V[x1][v1] - V[x][v])    # update value function using the Q-learning update rule

            if done:
              break

        time+=1
        # if time%1000 == 0:
        #     print(f"Episode: {time}")

    # print(V)

    with open('V'+str(i), 'wb') as file:
        pickle.dump(V, file)

    return V