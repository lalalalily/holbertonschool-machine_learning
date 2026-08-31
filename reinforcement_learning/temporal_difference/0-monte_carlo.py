#!/usr/bin/env python3
"""Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                 alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm"""
    for ep in range(episodes):
        state, _ = env.reset()
        episode = []
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break
        episode = np.array(episode, dtype=int)
        G = 0
        states = episode[:, 0]
        for i, (state, reward) in enumerate(episode[::-1]):
            G = reward + gamma * G
            idx = len(episode) - 1 - i
            if state not in states[:idx]:
                V[state] = V[state] + alpha * (G - V[state])
    return V
