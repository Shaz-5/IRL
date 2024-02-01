import numpy as np
import random
from grid_world_5x5 import *
import matplotlib.pyplot as plt

# 0 - Left ; 1 - Down ; 2 - Right ; 3 - Up

# Agent trained for 20 trajectories
sample_grid = MyGridWorld()
sample_grid_trainer = MyGridWorldTrainer()
sample_trajectories = sample_grid_trainer.all_in_one(sample_grid, 20)     # training (Q Learning)

print('Policy: \n')
for direction in sample_grid_trainer.matrix:
    print(direction)

print('\nPolicy (directions): \n')
for row in sample_grid_trainer.DirectionalMatrix:
    print(row)

print('\nQ value matrix: \n')
for row in sample_grid_trainer.Q:   # Q value for each state action pair
    print(row)

print('\nTrajectories: \n')
for trajectory in sample_trajectories:
    print(trajectory)

# function for plotting reward function
def plot_reward_function(reward, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = y = np.arange(0, 5, 1)
    X, Y = np.meshgrid(x, y)
    zs = reward  # True reward function
    Z = zs.reshape(X.shape)

    ax.view_init(30, -135)
    ax.plot_surface(X, Y, Z, alpha=0.5, cmap='tab10', rstride=1, cstride=1, edgecolors='k', lw=1)

    ax.set_title(title)

    plt.show()

# plot true reward function
plot_reward_function(sample_grid_trainer.env.RewardGrid, 'True Reward Function')

# function to generate transition matrix

def construct_transition_matrix(size=5, states=25, actions=4, noisy_move_chance=0.3):
    transition_matrix = np.zeros([states, states, actions])

    for state in range(states):
        for action in range(actions):
            i = int(state / size)
            j = state % size

            prob_stay = 1     # probability of staying in the current state

            # updating transition probabilities based on the chosen action
            if action == 0:  # Left
                if 0 < j:    # not at the left edge of the grid
                    prob_stay = prob_stay - (1 - noisy_move_chance)   # (1 - noisy_move_chance) = probability of a deterministic transition
                    j2 = j - 1      # update col index
                    transition_matrix[state][int(i * size + j2)][action] = 1 - noisy_move_chance    # int(i * size + j2) = index of the resulting state

            elif action == 1:  # Down
                if i < size - 1:
                    prob_stay = prob_stay - (1 - noisy_move_chance)
                    i2 = i + 1
                    transition_matrix[state][int(i2 * size + j)][action] = 1 - noisy_move_chance

            elif action == 2:  # Right
                if j < size - 1:
                    prob_stay = prob_stay - (1 - noisy_move_chance)
                    j2 = j + 1
                    transition_matrix[state][int(i * size + j2)][action] = 1 - noisy_move_chance

            elif action == 3:  # Up
                if 0 < i:
                    prob_stay = prob_stay - (1 - noisy_move_chance)
                    i2 = i - 1
                    transition_matrix[state][int(i2 * size + j)][action] = 1 - noisy_move_chance

            # updating transition probabilities based on the noisy actions (stochastic behavior)
            if 0 < j:
                prob_stay = prob_stay - (noisy_move_chance / 4)       # since 4 possible actions
                j2 = j - 1
                transition_matrix[state][int(i * size + j2)][action] += (noisy_move_chance / 4)

            if i < size - 1:
                prob_stay = prob_stay - (noisy_move_chance / 4)
                i2 = i + 1
                transition_matrix[state][int(i2 * size + j)][action] += (noisy_move_chance / 4)

            if j < size - 1:
                prob_stay = prob_stay - (noisy_move_chance / 4)
                j2 = j + 1
                transition_matrix[state][int(i * size + j2)][action] += (noisy_move_chance / 4)

            if 0 < i:
                prob_stay = prob_stay - (noisy_move_chance / 4)
                i2 = i - 1
                transition_matrix[state][int(i2 * size + j)][action] += (noisy_move_chance / 4)

            # probability of staying too small
            if prob_stay < 10**-15:
                prob_stay = 0
            transition_matrix[state][state][action] = prob_stay

    return transition_matrix

# 'Given' optimal policy
optimum_policy = [[2, 2, 2, 2, 2],
                 [3, 2, 2, 3, 3],
                 [3, 3, 3, 3, 3],
                 [3, 3, 2, 3, 3],
                 [3, 2, 2, 2, 3]]

# Directional matrix of given optimal policy
directional_matrix = []

for i in range(5):
    row = []
    for j in range(5):
        if optimum_policy[i][j] == 0:
            row.append('\u2190')  # Left arrow
        elif optimum_policy[i][j] == 1:
            row.append('\u2193')  # Down arrow
        elif optimum_policy[i][j] == 2:
            row.append('\u2192')  # Right arrow
        elif optimum_policy[i][j] == 3:
            row.append('\u2191')  # Up arrow
    directional_matrix.append(row)

print('Given Optimal Policy: \n')
for row in directional_matrix:
    print(row)

from scipy.optimize import linprog    # for linear programming

# linear programming approach for solving the inverse reinforcement learning problem (returns the reward function for given policy)

def perform_inverse_reinforcement_learning(policy, gamma=0.5, l1=10):
    trans_probs = construct_transition_matrix(size=5, states=25, actions=4, noisy_move_chance=0.3)
    conditions = []
    c = np.zeros([3 * 25])    # coefficients

    for i in range(25):
        optimal_action = policy[i]      # a1
        temp_trans_prob_matrix = gamma * trans_probs[:, :, optimal_action]      # γ⋅Pa1
        temp_inverse = np.linalg.inv(np.identity(25) - temp_trans_prob_matrix)  # (I−γ⋅Pa1​)^−1

        for j in range(4):
            if j != optimal_action:
                condition = -np.dot(trans_probs[i, :, optimal_action] - trans_probs[i, :, j], temp_inverse)   # (Pa1​−Pa​)(I−γ⋅Pa1​)^−1
                conditions.append(condition)

    equality = np.zeros(625)    # 625 equality constraints in the lp problem
    c[25:2 * 25] = -1           # ensure that rewards for non-optimal actions are negative
    c[2 * 25:] = l1             # regularization term

    conditions = np.array(conditions)   # contains coefficients
    conditions = np.reshape(conditions, [625, 75])      # 75 coefficients
    print(len(c), conditions.shape)
    rewards = linprog(c, A_ub=conditions, b_ub=equality)

    return rewards

policy = np.reshape(sample_grid_trainer.matrix,[25,1])
# policy = np.reshape(optimum_policy,[25,1])

reward = perform_inverse_reinforcement_learning(policy,gamma=0.5,l1=1)
reward = reward['x'][:25]
reward = np.reshape(reward,[5,5])

plot_reward_function(reward)    # we get a degenerate reward function

from cvxopt import matrix, solvers

# function to set up the constraints and variables for the linear programming problem

def initialize_solver_matrix(penalty=10):
    # For all states and all possible non-optimal actions to all states

    A = np.zeros([25**2, 3 * 25])     # linear inequality constraints
    b = np.zeros([25**2])             # right-hand side values for the linear constraints
    x = np.zeros([3 * 25])            # variables

    def initialize():
        size = 25
        num = 150   #offset
        i = 0

        while i < 25:
            A[num + i, i] = 1
            A[num + size + i, i] = -1

            j = 2
            while j < 4:
                A[num + j * size + i, i] = 1
                A[num + j * size + i, 2 * size + i] = -1
                j += 1

            b[num + i] = 1
            b[num + size + i] = 0
            i += 1

    initialize()

    x[25:] = -1
    x[-25:] = penalty

    return A, b, x

# optimization step for inverse reinforcement learning
# aims to maximize the rewards associated with optimal actions and penalize non-optimal actions

def perform_optimized_IRL(policy, gamma=0.5, penalty=10):
    TransitionMatrix = construct_transition_matrix()

    A, b, x = initialize_solver_matrix(penalty)
    i = 0

    while i < 25:
        optimalAction = int(policy[i])
        tempTransProbMatrix = gamma * TransitionMatrix[:, :, optimalAction]           # γ⋅Pa1
        patialInvertedMatrix = np.linalg.inv(np.identity(25) - tempTransProbMatrix)   #(I−γ⋅Pa1​)^−1

        temp = 0
        j = 0

        while j < 4:
            if j != optimalAction:      # penalize non-optimal actions
                otherPartialMatrix = TransitionMatrix[i, :, optimalAction] - TransitionMatrix[i, :, j]  # Pa1​−Pa
                val = -np.dot(otherPartialMatrix, patialInvertedMatrix)       # (Pa1​−Pa​)(I−γ⋅Pa1​)^−1
                pos = 25 * 3                        # starting index for the additional set of constraints related to the penalty term
                A[i * 3 + temp, :25] = val          # constraints associated with the reward vector for state i and the observed optimal action
                A[pos + i * 3 + temp, :25] = val    # constraints associated with the penalty term.
                A[pos + i * 3 + temp, 25 + i] = 1   # additional constraint to enforce the penalty term in the linear programming problem.
            else:
                temp = temp - 1
            temp = temp + 1
            j = j + 1
        i = i + 1

    x = matrix(x)
    A = matrix(A)
    b = matrix(b)

    return solvers.lp(x, A, b)

# perform IRL to obtain reward function

policy = np.reshape(optimum_policy, [25, 1])

rewards = perform_optimized_IRL(policy, gamma=0.1, penalty=1.05)
rewards = rewards['x']
rewards = rewards[:5 * 5]
rewards = rewards / max(rewards)      # normalize the rewards
rewards = np.reshape(rewards, [5, 5])

true = np.abs(sample_grid_trainer.env.RewardGrid)   # true rewards
obtained = np.abs(rewards)                          # estimated rewards
errors = true - obtained

error = np.abs(np.sum(np.sum(errors)))
print('\nError: \n',error)

plot_reward_function(rewards, 'Inverse RL Reward Function')

policy = np.reshape(sample_grid_trainer.matrix, [25, 1])

rewards = perform_optimized_IRL(policy, gamma=0.1, penalty=2.5)
rewards = rewards['x']
rewards = rewards[:5 * 5]
rewards = rewards / max(rewards)
rewards = np.reshape(rewards, [5, 5])

plot_reward_function(rewards, 'Inverse RL Reward Function (from expert)')
