#!/usr/bin/env python3
"""TD(lambtha) algorithm for value estimation."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambtha) algorithm to estimate a value function.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next
            action to take
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V, the updated value estimate
    """
    for ep in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros_like(V)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]

            eligibility[state] += 1.0
            eligibility *= lambtha * gamma

            V += alpha * delta * eligibility

            state = next_state

            if terminated or truncated:
                break

    return V
