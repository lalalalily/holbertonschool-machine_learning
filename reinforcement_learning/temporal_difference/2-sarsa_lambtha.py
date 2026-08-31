#!/usr/bin/env python3
"""SARSA(λ) module."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Uses epsilon-greedy to determine the next action."""
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Performs SARSA(λ).

    Args:
        env: the environment instance
        Q: numpy.ndarray of shape (s,a) containing the Q table
        lambtha: the eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q, the updated Q table
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        eligibility_trace = np.zeros_like(Q)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])

            eligibility_trace[state, action] += 1

            Q += alpha * delta * eligibility_trace
            eligibility_trace *= gamma * lambtha

            if terminated or truncated:
                break

            state, action = next_state, next_action

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * episode))

    return Q
