import gym
import numpy as np
import random, copy
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from IPython.display import Image
from math import exp, sqrt, pi
  
  
env = gym.make('MountainCar-v0')
env.seed(42)
e = env.env
discretization = 120

xmin = e.min_position                         # min value for position
xmax = e.max_position                         # max value for position
x_binsize = (xmax - xmin) / discretization    # size of each bin in the discretized space

vmin = -1 * e.max_speed                       # min value for velocity
vmax = e.max_speed                            # max value for velocity
v_binsize = (vmax - vmin) / discretization


# function to take a continuous state 's' from the environment and discretize it into a grid position
def get_state(s):
    xpos = min(discretization - 1, round((s[0] - xmin) / x_binsize))    # s[0] and s[1] - position and velocity of the agent
    vpos = min(discretization - 1, round((s[1] - vmin) / v_binsize))
    return int(xpos), int(vpos)     # indices in the discretized state space


# training agent using Q Learning
def Q_learning_train(env, discretization, epsilon=0.1, alpha=0.1, gamma=0.9, max_episodes=20000, 
                     save=False, video_path='./'):
#     time = 0
    min_steps = env._max_episode_steps + 1   # Minimum steps needed (initialized to maximum possible steps)
    
    # initialize Q-table with zeros
    Q = [[[0 for _ in range(env.action_space.n)] for _ in range(discretization)] for _ in range(discretization)]
    
    Q_optim = copy.deepcopy(Q)               # copy of Q to store the optimized Q-values later
    
    if save==True:
        env = gym.wrappers.RecordVideo(env, video_path, episode_trigger = lambda x: x % 2000 == 0)

    for time in tqdm(range(max_episodes), desc='Training agent..'):
#     while time < max_episodes: # or min_steps > 260:
        obs = env.reset()
        score = 0
        steps = 0

        greed_pol = random.randint(0, 9)

        while True:  # episode
            steps += 1
            
            if save==False and time % 2000 == 0:
                env.render()

            x, v = get_state(obs)
            a = random.randint(0, 2)

            # xploration or exploitation based on epsilon-greedy strategy
            if greed_pol == 0 or random.random() < 1 - epsilon:
                a = np.argmax(np.array(Q[x][v]))

            obs, R, done, info = env.step(a)

            x1, v1 = get_state(obs)

            score += R

            # Q(s, a) ← Q(s, a) + α ( R + γ * max{a'} * Q(s', a') - Q(s, a)) Bellman Eq
            Q[x][v][a] += alpha * (R + gamma * max(Q[x1][v1]) - Q[x][v][a])

            if done:  # episode terminated
                break

#         time += 1
        if time % 2000 == 0:
            print(f"Episode: {time} \t Steps: {steps} \t Score: {score}")

        # update the lowest steps and Q values
        if greed_pol == 0 and steps < min_steps:
            min_steps = steps
            Q_optim = copy.deepcopy(Q)
            
        env.close()

    return Q_optim
    
    
# function to render the trained Q-learning agent's performance in the environment
def render_agent(env, Q, N=1, render_filename='Render'):
    
    for i in range(N):
        obs = env.reset()

        images = []
        while True:
            img = env.render(mode='rgb_array')
            images.append(img)

            x, v = get_state(obs)
            a = np.argmax(np.array(Q[x][v]))
            obs, R, done, info = env.step(a)

            if done:
                break

        imageio.mimsave(f'{render_filename} {i+1}.gif', images, fps=30, loop=0)
        env.close()
        with open(f'{render_filename} {i+1}.gif', 'rb') as f:
            display(Image(data=f.read(), format='gif'))
            

# function for calculating the probability density function (PDF) of a normal distribution
def get_pdf(x, mean, std_dev):
    
    z = (x - mean) / std_dev
    pdf = (exp(-(z**2) / 2) / (sqrt(2 * pi) * std_dev))
    
    return pdf
    
    
# calculate the value function through Q Learning
def get_value_function(Q, mean, scale, i, alpha=0.1, gamma=0.99, epsilon=0.05, max_episodes=10000, save_path='./'):

    V = [[0 for _ in range(120)] for _ in range(120)]

    for episode in tqdm(range(max_episodes), desc='Calculating Value Function..'):
        obs = env.reset()

        while True:
            x, v = get_state(obs)
            a=np.argmax(np.array(Q[x][v]))

            obs, R, done, info = env.step(a)
            R = calc_pdf(x, mean, scale)  # calculate a reward using a probability density function (PDF)

            x1, v1 = get_state(obs)  # get next state (x1, v1)

            V[x][v] += alpha * (R + gamma * V[x1][v1] - V[x][v])  # update value function using the Q-learning update rule

            if done:
                break

    # Save the value function
    if save_path:
        index_str = str(i).zfill(2)
        with open(f'{save_path}{index_str}.pkl', 'wb') as file:
            pickle.dump(V, file)

    return V
