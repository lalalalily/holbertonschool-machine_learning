#!/usr/bin/env python3
"""TD(λ) module."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the TD(λ) algorithm.

    Args:
        env: the environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: a function that takes in a state and returns the next
            action to take
        lambtha: the eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate

    Returns:
        V, the updated value estimate
    """
    for episode in range(episodes):
        state, _ = env.reset()
        eligibility_trace = np.zeros_like(V)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]

            eligibility_trace[state] += 1

            V += alpha * delta * eligibility_trace
            eligibility_trace *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state

    return V
