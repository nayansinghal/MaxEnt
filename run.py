import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import product

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
	For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectorie
	\s

	trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
	horizon: the finite time horizon (int) of the problem for calculating state frequencies
	r_weights: a size F array of the weights of the current reward function to evaluate
	state_features: an S x F array that lists F feature values for each state in S

	return: an S x A policy in which each entry is the probability of taking action a in state s
	"""
	n_states = np.shape(trans_mat)[0]
	n_actions = np.shape(trans_mat)[1]

	partition = np.zeros((n_states, 1))
	policy = np.zeros((n_states, n_actions))

	partition[n_states-1] = 1
	reward = np.exp(np.dot(r_weights, state_features.T))

	# Calculate partition function for each state and policy value for (state,action)
	for i in range(horizon):
		new_partition = partition
		for state in range(n_states):
			for action in range(n_actions):
				p = np.array([trans_mat[state, action, new_state]*reward[state]*partition[new_state] for new_state in range(n_states)])
				policy[state, action] = np.sum(p)
			new_partition[state] = np.sum(policy[state, :])
			if state == n_states-1:
				new_partition[state] = 1
		partition = new_partition


	# Local action probability computation
	for state in range(n_states):
		for action in range(n_actions):
			if partition[state] != 0.0:
				policy[state, action] = policy[state, action] / partition[state]

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
	n_states = np.shape(trans_mat)[0]
	n_actions = np.shape(trans_mat)[1]

	# Copy start_dist for complete horizon, states x horizon
	exp_svf = np.zeros((n_states, horizon))
	exp_svf[0,0] = 1.0

	# Calculate Expected State Frequency for (state, time) pair
	for time in range(1, horizon):
		for state, action, new_state in product(range(n_states), range(n_actions), range(n_states)):
			exp_svf[new_state, time] += (exp_svf[state, time-1]* policy[state, action]* trans_mat[state, action, new_state])

	state_freq = exp_svf.sum(axis=1)
	return state_freq

def find_feature_expectations(state_features, demos):
	"""
	Compute expected feature expectations from demonstration trajectories

	:param state_features: an S x F array that lists F feature values for each state in S
	:param demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
	:return: expected feature expectation vector of size (state_features x 1)
	"""

	feature_exp = np.zeros(state_features.shape[1])
	for demo in demos:
		for state in demo:
			feature_exp += state_features[state]

	# Expected feature_exp
	return feature_exp/np.shape(demos)[0]

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
	feature_exp = find_feature_expectations(state_features, demos)

	n_states = np.shape(trans_mat)[0]
	n_actions = np.shape(trans_mat)[1]

	n_features = np.shape(state_features)[1]
	r_weights = np.zeros(n_features) + seed_weights

	# Probability for initial state trajectories
	start_state_count = np.zeros(n_states)
	for demo in demos:
		start_state_count[demo[0]] += 1
		p_start_dist = start_state_count / np.shape(demos)[0]

	# Iterate
	for epoch in range(n_epochs):
		# print("epoch: {}".format(epoch))

		# Calculate Max Ent Policy
		policy = calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features)

		# Calculate Expected State Frequency
		expected_svf = calcExpectedStateFreq(trans_mat, horizon, p_start_dist, policy)

		# Update reward weights using gradient
		gradient = feature_exp - expected_svf.T.dot(state_features)
		r_weights += learning_rate * gradient
		print epoch, np.linalg.norm(gradient)

	print policy
	print policy.argmax(axis=1)
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
	reward = []
	for demo in demos:

		sum = 0.0
		for i in demo:
			if i != 25:
				sum += reward_fxn[i]
		reward.append(sum)
	print reward
	reward_fxn = np.reshape(reward_fxn, (5, 5))
	print 'reward function', reward_fxn

	# Plot reward function
	plt.close("all")
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	X = np.arange(0, 5, 1)
	Y = np.arange(0, 5, 1)
	X, Y = np.meshgrid(X, Y)
	surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
						   linewidth=0, antialiased=False)
	plt.show()