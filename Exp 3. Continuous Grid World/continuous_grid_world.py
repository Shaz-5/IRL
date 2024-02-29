import numpy as np
import random, copy
from math import exp,sqrt,pi
import seaborn as sns
import matplotlib.pyplot as plt


# Continuous Grid World Environment
class ContinousGridWorld:
    
    def __init__(self, N=50, region=0.8):
        self.action_space = 4
        self.observation_space = ([0,1],[0,1])
        self.N = N
        self.reward_region = round(region* N)
        self.R = [[1.0 if i >= self.reward_region and j >= self.reward_region else 0.0 for j in range(N)] for i in range(N)]
        
    def get_reward(self):
        return self.R
    
    def reset(self):
        self.__init__()
    
    def plot_reward(self, title='True Reward', filename=None):
        reward = self.R
        plt.figure(figsize=(10,8))
        sns.heatmap(reward, xticklabels=False, yticklabels=False, cmap='cividis')
        plt.title(title)
        plt.xlabel("States in x")
        plt.ylabel("States in y")
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.2)
        plt.show()
        
    # move agent 0.2 in intended direction with addition of uniform noise in [-0.1,1]
    def move(step, i, j, a):

        movements = {"L": (0, -0.2), "R": (0, 0.2), "U": (-0.2, 0), "D": (0.2, 0)}

        # move agent with added noise in [-1,1]
        di, dj = movements.get(a, (0, 0))
        i += di + random.uniform(-0.1, 0.1)
        j += dj + random.uniform(-0.1, 0.1)

        # check if the new state is an absorbing state
        absorbing_state = i >= 0.8 and j >= 0.8

        # handle boundary conditions
        i = max(0.0, min(1.0, i))
        j = max(0.0, min(1.0, j))

        return i, j, absorbing_state
    
    
# Class with Functions to calculate Value Functions and Policies
class ContinuousGridWorldSolver:
    
    def __init__(self, N=50, region=0.8, error=0.1, move_length=0.2, gamma=0.9):
        self.N = N                                     # discretization
        self.reward_region = round(region * N)         # region where reward is 1
        self.error = round(error * N)                  # noise
        self.move_length = round(move_length * N)      # length of a move or step in the environment
        self.gamma = gamma
        self.R = [[0.0 for i in range(N)] for j in range(N)]
        
        
    # function to calculate the expected value for a given state-action pair
    def get_expected_value(self, V, i, j, a):

        # number of iterations for averaging, considering the error range
        averaging_over = self.error * 2 + 1
        avg = 0.0

        # if the action is to move left
        if a == "L":
            # iterate over possible column indices within the specified error range to the left
            for k in range(j - self.move_length - self.error, j - self.move_length + self.error + 1):
                move = max(0, k)  # boundary conditions (move cannot be less than 0)

                # calculate the average value considering the reward and the discounted value of the next state
                r = self.R[i][j]
                avg += r + self.gamma * V[i][move]

        elif a == "R":
            for k in range(j + self.move_length - self.error, j + self.move_length + self.error + 1):
                move = min(self.N - 1, k)  # handle boundary conditions (move cannot exceed N-1)

                r = self.R[i][j]
                avg += r + self.gamma * V[i][move]

        elif a == "U":
            for k in range(i - self.move_length - self.error, i - self.move_length + self.error + 1):
                move = max(0, k)  # handle boundary conditions (move cannot be less than 0)

                r = self.R[i][j]
                avg += r + self.gamma * V[move][j]

        else:
            for k in range(i + self.move_length - self.error, i + self.move_length + self.error + 1):
                move = min(self.N - 1, k)  # handle boundary conditions (move cannot exceed N-1)

                r = self.R[i][j]
                avg += r + self.gamma * V[move][j]

        # return averaged value divided by the total number of iterations
        return avg / averaging_over

    
    # function to update state values based on the given policy
    def update_values(self, V, policy):
        updated_values = [[0.0 for i in range(self.N)] for j in range(self.N)]

        for i in range(self.N):
            for j in range(self.N):
                # update state value using the expected value function and the current policy
                updated_values[i][j] = self.get_expected_value(V, i, j, policy[i][j])

        return updated_values
    
    
    # function to update the policy based on the state values
    def update_policy(self, V):
        updated_policy = [["U" for _ in range(self.N)] for _ in range(self.N)]

        for i in range(self.N):
            for j in range(self.N):
                # initialize action with "U"
                best_action = "U"
                best_expected_value = self.get_expected_value(V, i, j, "U")

                # iterate over possible actions ("D", "L", "R")
                for action in ["D", "L", "R"]:
                    tmp_expected_value = self.get_expected_value(V, i, j, action)

                    # update the action with the maximum expected value for the state
                    if tmp_expected_value > best_expected_value:
                        best_expected_value = tmp_expected_value
                        best_action = action

                updated_policy[i][j] = best_action

        return updated_policy
   
    
    # function to get optimal policy using policy iteration
    def get_optimal_policy(self, discretization, rewards_matrix=None):

        # update grid size and rewards matrix if provided
        N = discretization
        if rewards_matrix is not None:
            R = rewards_matrix
            self.R = R
        else:
            # if rewards matrix is not provided, set rewards in the bottom-right region to 1.0
            R = [[1.0 if i >= self.reward_region and j >= self.reward_region else 0.0 for j in range(N)] for i in range(N)]
            self.R = R

        # initialize state-value function (V) and policy matrix with all actions set to "D" (Down)
        V = [[0.0 for _ in range(N)] for _ in range(N)]
        policy = [["D" for _ in range(N)] for _ in range(N)]  # U:0, D:1, L:2, R:3

        t = 0
        while True:
            t += 1

            tt = 0
            # policy evaluation
            while True:
                tt += 1
                _V = self.update_values(copy.deepcopy(V), policy)

                # check if the state-value function has converged
                has_converged = all(abs(V[i][j] - _V[i][j]) <= 0.001 for i in range(N) for j in range(N))
                if has_converged:
                    break
                else:
                    V = copy.deepcopy(_V)

            # policy improvement
            _policy = self.update_policy(V)

            # check if policy has changed; if not, optimal policy found
            if policy == _policy:
                break
            else:
                policy = copy.deepcopy(_policy)

        return policy, V
    
    
    # function to compare two policies and calculate the similarity
    def compare_policies(self, policy_set1, policy_set2):
        matching_states = sum(set(policy_set1[i][j]) == set(policy_set2[i][j]) for i in range(self.N) for j in range(self.N))
        total_states = self.N * self.N
        similarity_ratio = matching_states / total_states
        return similarity_ratio
    
    
    # functions to visualize value function
    def plot_value_function(self, V, title='Value Function', filename=None):
        plt.figure(figsize=(10,8))
        sns.heatmap(V, xticklabels=False, yticklabels=False, cmap='cividis')
        plt.title(title)
        plt.xlabel("States in x")
        plt.ylabel("States in y")
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.2)
        plt.show()
        
    def show_value_function(self, V, title='Value Function', filename=None):
        plt.figure(figsize=(20,20))
        sns.heatmap(V, annot=True,fmt='.1f', xticklabels=False, 
                    yticklabels=False, cmap='gray_r', cbar=False, vmax=0, vmin=0)
        plt.title(title)
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.2)
        plt.show()
    
    # functions to visualize reward function
    def plot_reward_function(self, reward, title='Reward', filename=None):
        plt.figure(figsize=(10,8))
        sns.heatmap(reward, xticklabels=False, yticklabels=False, cmap='cividis')
        plt.title(title)
        plt.xlabel("States in x")
        plt.ylabel("States in y")
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.2)
        plt.show()
        
    def show_reward_function(self, reward, title='Reward', filename=None):
        plt.figure(figsize=(20,20))
        sns.heatmap(reward, annot=True,fmt='.1f', xticklabels=False, 
                    yticklabels=False, cmap='gray_r', cbar=False, vmax=0, vmin=0)
        plt.title(title)
        plt.xlabel("States in x")
        plt.ylabel("States in y")
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.2)
        plt.show()
        
    
    # function to visualize policy
    def plot_policy_matrix(self, policy, title='Policy', grid=False):
        
        def policy_to_arrows(policy):
            arrow_mapping = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
            arrows = [[arrow_mapping[action] for action in row] for row in policy]
            return arrows
        
        matrix = np.array(policy).reshape(self.N,self.N)
        matrix = policy_to_arrows(matrix)
        fig, ax = plt.subplots(figsize=(9.5,9.5))
        cax = ax.matshow([[ord(cell) for cell in row] for row in matrix], cmap='gray_r', vmin=0, vmax=0)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                ax.text(j, i, matrix[i][j], va='center', ha='center', fontsize=10, color='k')
        ax.set_xticks([])
        ax.set_yticks([])
        if grid:
            length = self.N
            plt.hlines(y=np.arange(0, length)+0.5, xmin=np.full(length, 0)-0.5, xmax=np.full(length, length)-0.5, color="black")
            plt.vlines(x=np.arange(0, length)+0.5, ymin=np.full(length, 0)-0.5, ymax=np.full(length, length)-0.5, color="black")
        plt.title(title)
        plt.show()