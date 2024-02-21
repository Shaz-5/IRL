import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from IPython.display import Image
import seaborn as sns

"""Actions:
* 0 - Left
* 1 - Down
* 2 - Right
* 3 - Up

Agent Position:  1.1
"""

# Defining GridWorld Environment Class

class GridWorld:

    # initialize the environment with default values
    def __init__(self, size=5, noisyMoveChance=0.3, EnableNoise=True):
        self.basic_reset()
        self.EnableNoise = EnableNoise
        if 0 < size:
            self.size = int(size)
            self.RewardGrid = np.zeros([size, size])
            self.RewardGrid[0][size-1] = 1
            self.PositionGrid = np.zeros([size, size])
            self.PositionGrid[size-1][0] = 1.1
            self.observation_spaces = self.size * self.size
            self.currI = size-1
            self.currJ = 0
            self.observation_spaces = self.size * self.size
        if 0 < noisyMoveChance < 1:     # noisy probability value
            self.noisyMoveChance = noisyMoveChance

    # resets the environment to its initial state
    def basic_reset(self):
        self.size = 5                                    # 5x5 grid
        self.RewardGrid = np.zeros([5, 5])               # grid representing rewards
        self.RewardGrid[0][4] = 1                        # sets reward in the top-right cell to 1
        self.PositionGrid = np.zeros([5, 5])             # grid representing the current position of the agent
        self.PositionGrid[4][0] = 1.1                    # sets agent's initial position in the bottom-left cell
        self.action_space = 4                            # no. of possible actions
        self.noisyMoveChance = 0.3                       # probability of noisy move
        self.currI = 4                                   # row index
        self.currJ = 0                                   # col index
        self.DoneStatus = False                          # whether the episode is terminated
        self.EnableNoise = True                          # enable or disable noise
        self.observation_spaces = self.size * self.size  # total no. of observations

    # reset environment with parameters
    def reset(self, size=5, noisyMoveChance=0.3, EnableNoise=True):
        self.__init__(size, noisyMoveChance, EnableNoise)
        return self.currI * self.size + self.currJ         # current state of the agent

    # print the reward grid
    def print_reward_grid(self):
        print(self.RewardGrid)
        print()
        
    # plot reward function
    def plot_reward_function(self, reward, title='', filename=None):

        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')

        x = y = np.arange(1,6)
        X, Y = np.meshgrid(x, y)
        zs = reward
        Z = zs.reshape(X.shape)
        ax.view_init(30, -135)
        ax.set_xticks(range(1,6))
        ax.set_xticklabels(range(1,6))
        ax.set_yticks(range(1,6))
        ax.set_yticklabels(range(1,6))
        ax.plot_surface(X, Y, Z, alpha=0.5, cmap='tab10', rstride=1, cstride=1, edgecolors='k', lw=1)

        ax.set_title(title)
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.2)
        plt.show()

    # print the position grid
    def print_position_grid(self):
        print(self.PositionGrid)
        print()
        
    # render environment
    def render(self):
        PositionGrid = self.get_position_grid()
        agent_position = np.argwhere(PositionGrid == 1.1)[0]
        
        fig, ax = plt.subplots(figsize=(3,3))
        ax.plot(agent_position[1] + 0.5, self.size - 1 - agent_position[0] + 0.5, 'kh', markersize=20)
        ax.set_xticks(range(self.size+1), labels=[])
        ax.set_yticks(range(self.size+1), labels=[])
        ax.tick_params(axis='x', which='major', tick1On=False, tick2On=False)
        ax.tick_params(axis='y', which='major', tick1On=False, tick2On=False)
        ax.grid()
        plt.tight_layout()
        plt.show()
        
    # return position grid
    def get_position_grid(self):
        return self.PositionGrid

    # return no. of actions
    def get_available_moves(self):
        return self.action_space

    # return size of grid
    def get_size(self):
        return self.size
        
    # return observations of grid
    def get_observations(self):
        return self.observation_spaces

    # takes an action and updates the agent's position
    def move(self, action):
        rand_num = random.random()
        if self.EnableNoise and rand_num <= self.noisyMoveChance:
            self.make_noisy_move(action)
        else:
            self.make_proper_move(action)
        return self.currI, self.currJ, self.currI * self.size + self.currJ, self.RewardGrid[self.currI][self.currJ], self.DoneStatus
        # x, y, next state, reward, done status
    
    # noisy move with random action
    def make_noisy_move(self, action):
        rand_num = random.randint(0, 3)
        self.make_proper_move(rand_num)

    # proper move based on given action
    def make_proper_move(self, action):
        if action == 0:  # Left
            if 0 < self.currJ:
                self.PositionGrid[self.currI][self.currJ] = 0
                self.currJ -= 1
                self.PositionGrid[self.currI][self.currJ] = 1.1

        elif action == 1:  # Down
            if self.currI < self.size - 1:
                self.PositionGrid[self.currI][self.currJ] = 0
                self.currI += 1
                self.PositionGrid[self.currI][self.currJ] = 1.1

        elif action == 2:  # Right
            if self.currJ < self.size - 1:
                self.PositionGrid[self.currI][self.currJ] = 0
                self.currJ += 1
                self.PositionGrid[self.currI][self.currJ] = 1.1

        elif action == 3:  # Up
            if 0 < self.currI:
                self.PositionGrid[self.currI][self.currJ] = 0
                self.currI -= 1
                self.PositionGrid[self.currI][self.currJ] = 1.1

        if self.currI == 0 and self.currJ == self.size - 1:   # termination condition reached
            self.DoneStatus = True

    # call move method on action and return output of it
    def step(self, action):
        return self.move(action)




# Define Q-Learning trainer class

class GridWorldTrainer:

    def __init__(self, env_model):
        self.env = env_model
        self.Q = np.zeros([self.env.observation_spaces, self.env.action_space])
        self.matrix = []
        self.DirectionalMatrix = []
        self.Trajectories = []
    
    # train using q learning
    def train_agent(self, episodes=20000, alpha = 0.6, gamma = 0.9):
        env = self.env
        Q = np.zeros([env.observation_spaces, env.action_space])

        for episode in tqdm(range(1, episodes+1), desc=f'Training agent for {episodes} episodes..'):
            done = False
            total_reward = 0
            state = env.reset()    # reset env

            while not done:
                if episode < 500:                    # exploration
                    action = random.randint(0, 3)
                else:
                    action = np.argmax(Q[state])     # exploitation
                    
                i, j, state2, reward, done = env.step(action)     # takes an action
                Q[state, action] += alpha * (reward + gamma * np.max(Q[state2]) - Q[state, action])  # update q value
                total_reward += reward
                state = state2

        self.Q = Q    # learned q values matrix
        return Q

    # get optimal directions from learned q values
    def get_policy(self, Q):
        matrix = []

        for i in range(0, self.env.size*self.env.size):
            matrix.append(np.argmax(Q[i]))      # appends the index of the action with maximum Q-value
        matrix = np.reshape(matrix, (self.env.size, self.env.size))

        self.matrix = matrix
        return matrix
    
    # get directional matrix of policy
    def get_directions(self,policy):
        matrix = policy
        DirectionalMatrix = []
        for i in range(self.env.size):
            row = []
            for j in range(self.env.size):
                if matrix[i][j] == 0:
                    row.append('\u2190')    # left symbol
                elif matrix[i][j] == 1:
                    row.append('\u2193')    # down symbol
                elif matrix[i][j] == 2:
                    row.append('\u2192')    # right symbol
                elif matrix[i][j] == 3:
                    row.append('\u2191')    # up symbol
            DirectionalMatrix.append(row)

        self.DirectionalMatrix = DirectionalMatrix
        return DirectionalMatrix

    # generate trajectories based on optimal actions
    def get_trajectories(self, matrix, num_trajectories):
        Trajectories = []

        for iters in range(num_trajectories):
            path = []       # list for a single trajectory
            done = False
            state = self.env.reset()
            total_reward = 0
            path.append(state)
            i = int(state / self.env.size)    # row index
            j = state % self.env.size         # col index

            # trajectory loop
            while not done:
                action = matrix[i][j]       # retrieve action
                i, j, state2, reward, done = self.env.step(action)    # take action
                total_reward += reward
                state = state2          # update state
                path.append(state)

            Trajectories.append(path)

        self.Trajectories = Trajectories
        return Trajectories
    
    # Print the Q Table
    def print_q_table(self):
        print(self.Q)
        print()
        
    # Print the Policy Matrix
    def print_policy_matrix(self):
        for row in self.matrix:
            print(row)
        print()
        
    # Print the Policy Matrix
    def print_policy_directional_matrix(self):
        for row in self.DirectionalMatrix:
            print(row)
        print()
        
    # Plot the Policy
    def plot_policy_directional_matrix(self, policy, title='Policy'):
        matrix = self.get_directions(policy)
        fig, ax = plt.subplots(figsize=(3,3))
        cax = ax.matshow([[ord(cell) for cell in row] for row in matrix], cmap='gray_r', vmin=0, vmax=0)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                ax.text(j, i, matrix[i][j], va='center', ha='center', fontsize=14, color='k')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, which='minor', color='k', linestyle='-', linewidth=2)
        length = 5
        plt.hlines(y=np.arange(0, length)+0.5, xmin=np.full(length, 0)-0.5, xmax=np.full(length, length)-0.5, color="black")
        plt.vlines(x=np.arange(0, length)+0.5, ymin=np.full(length, 0)-0.5, ymax=np.full(length, length)-0.5, color="black")
        plt.title(title)
        plt.show()
        
    # Print the Trajectories
    def print_trajectories(self):
        for trajectory in self.Trajectories:
            print(trajectory)
        print()
            
    def visualize_trajectories(self, filename=None):
        env_model = self.env
        trajectories = self.Trajectories
        num_trajectories = len(trajectories)
        num_cols = min(num_trajectories, 5)
        num_rows = (num_trajectories - 1) // num_cols + 1

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols+2, 2 * num_rows+2))
        axs = axs.flatten()

        for i in range(num_trajectories):
            ax = axs[i]
            ax.set_xticks(range(env_model.size), labels=[])
            ax.set_yticks(range(env_model.size), labels=[])
            ax.tick_params(axis='x', which='major', tick1On=False, tick2On=False)
            ax.tick_params(axis='y', which='major', tick1On=False, tick2On=False)
            ax.grid()

            # initial agent position
            state = trajectories[i][0]
            ii, jj = divmod(state, env_model.size)
            ax.plot(jj + 0.5, env_model.size - 1 - ii + 0.5, 'kh', markersize=20)  # agent position

            # plot agent's movements
            for t in range(1, len(trajectories[i])):
                next_state = trajectories[i][t]
                next_ii, next_jj = divmod(next_state, env_model.size)
                ax.arrow(jj + 0.5, env_model.size - 1 - ii + 0.5,
                         next_jj - jj, ii - next_ii,
                         head_width=0.1, head_length=0.1, fc='r', ec='r')    # position of arrows (of movement)
                ii, jj = next_ii, next_jj

            ax.set_title(f'Trajectory {i + 1}')


        plt.tight_layout()
        if filename:
        	plt.savefig(filename)
        plt.show()
    
    # visualize agent trajectory
    def render_trajectory(self, trajectory=0, filename=None, show=False):

        trajectories = self.Trajectories[trajectory]
        env_model = self.env
        
        fig, ax = plt.subplots(figsize=(4,4))

        state = trajectories[0]
        ii, jj = divmod(state, env_model.size)
        ax.plot(jj + 0.5, env_model.size - 1 - ii + 0.5, 'kh', markersize=20)

        def update(frame):
            ax.clear()

            state = trajectories[frame]
            ii, jj = divmod(state, env_model.size)
            ax.plot(jj + 0.5, env_model.size - 1 - ii + 0.5, 'kh', markersize=20)  # agent position
            ax.set_xticks(range(env_model.size+1), labels=[])
            ax.set_yticks(range(env_model.size+1), labels=[])
            ax.tick_params(axis='x', which='major', tick1On=False, tick2On=False)
            ax.tick_params(axis='y', which='major', tick1On=False, tick2On=False)
            ax.grid()
            ax.set_title(f'Trajectory {trajectory + 1}')

        rendered = FuncAnimation(fig, update, frames=len(trajectories), interval=300, repeat=True)
        
        if filename:
            rendered.save(filename)
            if show:
                display(Image(filename))

        if not filename:
            plt.show()
        plt.close()

    # all training functions
    def train_and_get_trajectories(self, env_model, num_trajectories, episodes=20000, alpha = 0.6, gamma = 0.9):
        self.env = env_model
        Q = self.train_agent(episodes, alpha, gamma)
        matrix = self.get_directions(Q)
        trajectories = self.get_trajectories(matrix, num_trajectories)
        return trajectories
