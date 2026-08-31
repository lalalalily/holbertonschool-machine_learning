#!/usr/bin/env python3
"""
Monte Carlo evaluation algorithm
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate value functions.

    Parameters:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        # Generate an episode
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))
            
            if terminated or truncated:
                break
            state = next_state

        # Backpropagate updates using First-Visit Monte Carlo
        G = 0
        visited_states = set()
        
        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            if state not in visited_states:
                visited_states.add(state)
                if alpha is not None:
                    V[state] = V[state] + alpha * (G - V[state])
                else:
                    # Fallback standard sample average if alpha is set to None
                    V[state] = V[state] + (1 / episodes) * (G - V[state])

    return V
