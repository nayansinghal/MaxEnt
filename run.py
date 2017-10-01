import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def build_trans_mat_gridworld():
    # 5x5 gridworld laid out like:
    # 0  1  2  3  4
    # 5  6  7  8  9
    # ...
    # 20 21 22 23 24
    # where 24 is a goal state that always transitions to a
    # special zero-reward terminal state (25) with no available actions
    trans_mat = np.zeros((26, 4, 26))

    # NOTE: the following iterations only happen for states 0-23.
    # This means terminal state 25 has zero probability to transition to any state,
    # even itself, making it terminal, and state 24 is handled specially below.

    # Action 0 = down
    for s in range(24):
        if s < 20:
            trans_mat[s, 0, s + 5] = 1
        else:
            trans_mat[s, 0, s] = 1

    # Action 1 = up
    for s in range(24):
        if s >= 5:
            trans_mat[s, 1, s - 5] = 1
        else:
            trans_mat[s, 1, s] = 1

    # Action 2 = left
    for s in range(24):
        if s % 5 > 0:
            trans_mat[s, 2, s - 1] = 1
        else:
            trans_mat[s, 2, s] = 1

            # Action 3 = right
    for s in range(24):
        if s % 5 < 4:
            trans_mat[s, 3, s + 1] = 1
        else:
            trans_mat[s, 3, s] = 1

    # Finally, goal state always goes to zero reward terminal state
    for a in range(4):
        trans_mat[24, a, 25] = 1

    return trans_mat


def calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features):
    """
    For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectories

    trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
    horizon: the finite time horizon (int) of the problem for calculating state frequencies
    r_weights: a size F array of the weights of the current reward function to evaluate
    state_features: an S x F array that lists F feature values for each state in S

    return: an S x A policy in which each entry is the probability of taking action a in state s
    """
    n_states = np.shape(trans_mat)[0]
    n_actions = np.shape(trans_mat)[1]

    Value = np.nan_to_num(np.ones((n_states, 1))*float("-inf"))
    diff = np.ones((n_states, 1))
    discount = 0.01

    while (diff > 1e-4).all():
        new_Value = Value;
        for action in range(n_actions):
            for states in range(n_states):
                new_Value[states] = max(new_Value[states], np.dot(r_weights, state_features[states]) + discount* np.sum(trans_mat[states, action, states_new]*Value[states_new] for states_new in range(n_states)))

        diff = abs(Value-new_Value)
        Value = new_Value

    policy = np.zeros((n_states, n_actions))
    for state in range(n_states):
        for action in range(n_actions):
            p = np.array([trans_mat[state, action, new_state] for new_state in range(n_states)])
            policy[state, action] = p.dot(np.dot(r_weights, state_features[states]) + discount * Value)

    # Softmax by row to interpret these values as probabilities.
    policy -= policy.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    policy = np.exp(policy)/np.exp(policy).sum(axis=1).reshape((n_states, 1))
    return policy


def calcExpectedStateFreq(trans_mat, horizon, start_dist, policy):
    """
    Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon

    trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
    horizon: the finite time horizon (int) of the problem for calculating state frequencies
    start_dist: a size S array of starting start probabilities - must sum to 1
    policy: an S x A array array of probabilities of taking action a when in state s

    return: a size S array of expected state visitation frequencies
    """

    state_freq = np.zeros(len(start_dist))
    return state_freq

def find_feature_expectations(state_features, demos):
    """
    Compute expected feature expectations from demonstraton trajectories

    :param state_features:
    :param demos:
    :return: expected feature expectation vector of size (state_features x 1)
    """

    feature_expectations = np.zeros(state_features.shape[1])
    for demo in demos:
        for state in demo:
            feature_expectations += state_features[state]

    return feature_expectations/np.shape(demos)[0]

def maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate):
    """
    Compute a MaxEnt reward function from demonstration trajectories

    trans_mat: an S x A x S' array that describes transition probabilites from state s to s' if action a is taken
    state_features: an S x F array that lists F feature values for each state in S
    demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
    seed_weights: a size F array of starting reward weights
    n_epochs: how many times (int) to perform gradient descent steps
    horizon: the finite time horizon (int) of the problem for calculating state frequencies
    learning_rate: a multiplicative factor (float) that determines gradient step size

    return: a size F array of reward weights
    """
    feature_expectations = find_feature_expectations(state_features, demos)

    n_features = np.shape(state_features)[1]
    r_weights = np.zeros(n_features)
    return r_weights


if __name__ == '__main__':

    # Build domain, features, and demos
    trans_mat = build_trans_mat_gridworld()
    state_features = np.eye(26, 25)  # Terminal state has no features, forcing zero reward
    demos = [[0, 1, 2, 3, 4, 9, 14, 19, 24, 25], [0, 5, 10, 15, 20, 21, 22, 23, 24, 25],
             [0, 5, 6, 11, 12, 17, 18, 23, 24, 25], [0, 1, 6, 7, 12, 13, 18, 19, 24, 25]]
    seed_weights = np.zeros(25)

    # Parameters
    n_epochs = 100
    horizon = 10
    learning_rate = 1.0

    # Main algorithm call
    r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate)

    # Construct reward function from weights and state features
    reward_fxn = []
    for s_i in range(25):
        reward_fxn.append(np.dot(r_weights, state_features[s_i]))
    reward_fxn = np.reshape(reward_fxn, (5, 5))

    # Plot reward function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(0, 5, 1)
    Y = np.arange(0, 5, 1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()