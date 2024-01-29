import numpy as np
import copy, random

# calculate the expected value for a given state-action pair
# it is the average cumulative reward, including the immediate reward and the discounted future rewards, for taking a specific action in a particular state

def get_expected_value(V, i, j, a):
    global N
    global reward_region
    global error
    global move_length
    global gamma
    global R

    # number of iterations for averaging, considering the error range
    averaging_over = error * 2 + 1
    avg = 0.0

    # if the action is to move left
    if a == "L":
        # iterate over possible column indices within the specified error range to the left
        for k in range(j - move_length - error, j - move_length + error + 1):
            move = k
            # handle boundary conditions
            if move < 0:
                move = 0

            # calculate the average value considering the reward and the discounted value of the next state
            r = R[i][j]
            avg += (r + gamma * V[i][move])

    # move right
    elif a == "R":

        for k in range(j + move_length - error, j + move_length + error + 1):
            move = k
            # boundary conditions
            if move >= N:
                move = N - 1

            # average value
            r = R[i][j]
            avg += (r + gamma * V[i][move])

    # move up
    elif a == "U":

        for k in range(i - move_length - error, i - move_length + error + 1):
            move = k

            if move < 0:
                move = 0

            r = R[i][j]
            avg += (r + gamma * V[move][j])

    # down
    else:

        for k in range(i + move_length - error, i + move_length + error + 1):
            move = k

            if move >= N:
                move = N - 1

            r = R[i][j]
            avg += (r + gamma * V[move][j])

    # return the averaged value divided by the total number of iterations
    return avg / averaging_over

# function to generate new state and associated reward based on the current state and action

def get_state_and_reward(i, j, a):

    if a == "U":
        # if action is "U" (up)
        i = i + random.randint(-1 * error, error)                 # add noiose for randomness
        j = j - move_length + random.randint(-1 * error, error)

    elif a == "D":
        # down
        i = i + random.randint(-1 * error, error)
        j = j + move_length + random.randint(-1 * error, error)

    elif a == "L":
        # left
        i = i - move_length + random.randint(-1 * error, error)
        j = j + random.randint(-1 * error, error)

    else:
        # right
        i = i + move_length + random.randint(-1 * error, error)
        j = j + random.randint(-1 * error, error)

    r = 0.0
    if i >= reward_region and j >= reward_region:
        r = 1.0

    # checking grid boundaries
    if i < 0:
        if j < 0:
            return 0, 0, r
        elif j >= N:
            return 0, N - 1, r
        else:
            return 0, j, r

    elif i >= N:
        if j < 0:
            return N - 1, 0, r
        elif j >= N:
            return N - 1, N - 1, r
        else:
            return N - 1, j, r

    else:
        if j < 0:
            return i, 0, r
        elif j >= N:
            return i, N - 1, r
        else:
            return i, j, r

# function to update state values based on the given policy

def update_values(V, policy):

    global N, reward_region, error, move_length, gamma, R

    # 2D array for updated state values
    v = [[0.0 for i in range(N)] for j in range(N)]

    # iterate over all states
    for i in range(N):
        for j in range(N):
            # update state value using the expected value function and the current policy
            v[i][j] = get_expected_value(V, i, j, policy[i][j])

    return v

# function to update the policy based on the state values

def update_policy(V):

    global N, reward_region, error, move_length, gamma, R

    # 2D array for updated policy
    p = [["U" for i in range(N)] for j in range(N)]

    # iterate over all states
    for i in range(N):
        for j in range(N):
            k = "U"
            maxim = get_expected_value(V, i, j, "U")

            # iterate over possible actions ("D", "L", "R")
            for x in ["D", "L", "R"]:
                tmp_exp = get_expected_value(V, i, j, x)
                # update the action with the maximum expected value for the state
                if tmp_exp > maxim:
                    maxim = tmp_exp
                    k = x
            p[i][j] = k

    return p

# function to get an equivalent policy based on the state values

def get_equivalent_policy(V):

    global N, reward_region, error, move_length, gamma, R

    #2D array for equivalent policy
    p = [["U" for i in range(N)] for j in range(N)]

    # iterate over all states
    for i in range(N):
        for j in range(N):
            k = {"U"}
            maxim = get_expected_value(V, i, j, "U")

            # iterate over possible actions ("D", "L", "R")
            for x in ["D", "L", "R"]:
                tmp_exp = get_expected_value(V, i, j, x)
                # update the set of equivalent actions that have similar expected values
                if abs(tmp_exp - maxim) < 0.0001:
                    maxim = max(tmp_exp, maxim)
                    k.add(x)
                elif tmp_exp > maxim:
                    maxim = tmp_exp
                    k = {x}
            p[i][j] = k

    return p

# function to display the state-value function

def show_value(Val):
    global N, reward_region, error, move_length, gamma, R

    print("STATE-VALUE FUNCTION")
    for i in range(N):
        for j in range(N):
            print("%.1f" % Val[i][j], end=" ")
        print()

# function to get optimal policy using policy iteration

def get_optimal_policy(n, r):

    global N, reward_region, error, move_length, gamma, R

    # update grid size and rewards matrix if provided
    N = n
    if r is not None:
        R = r
    else:
        # if rewards matrix is not provided, set rewards in the top-right region to 1.0
        for i in range(N):
            for j in range(N):
                if i >= reward_region and j >= reward_region:
                    R[i][j] = 1.0

    # initialize state-value function (V) and policy matrix with all actions set to "D" (Down)
    V = [[0.0 for i in range(N)] for j in range(N)]
    policy = [["D" for i in range(N)] for j in range(N)]  # U:0, D:1, L:2, R:3

    t = 0
    while 1 > 0:
        t += 1

        tt = 0
        # policy evaluation
        while 1 > 0:
            tt += 1
            _V = update_values(copy.deepcopy(V), policy)

            # check if state-value function has converged
            flag_break = True
            for i in range(N):
                for j in range(N):
                    if abs(V[i][j] - _V[i][j]) > 0.001:
                        flag_break = False
                        break
                if not flag_break:
                    break

            if flag_break:
                break
            else:
                V = copy.deepcopy(_V)

        # policy improvement
        _policy = update_policy(V)

        # check if policy has changed, if not, optimal policy found
        if policy == _policy:
            break
        else:
            policy = copy.deepcopy(_policy)

    # compute equivalent policy and return both optimal policy and its equivalent version
    equiv_policy = get_equivalent_policy(V)
    # show_value(V)
    return policy, equiv_policy

N=50
reward_region=round(0.8*N)
error=round(0.1*N)
move_length=round(0.2*N)
gamma=0.9
R=[[0.0 for i in range(N)] for j in range(N)]
