#!/usr/bin/env python3
"""Module for training a Monte-Carlo policy gradient agent."""
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Trains a policy using the Monte-Carlo policy gradient algorithm.

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: learning rate
        gamma: discount factor

    Returns:
        scores: list containing cumulative rewards for each episode
    """
    # CartPole state dimension is 4, action space size is 2
    weight = np.random.rand(4, 2)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        state = state.reshape(1, -1)

        gradients = []
        rewards = []

        done = False
        truncated = False

        while not (done or truncated):
            action, grad = policy_gradient(state, weight)
            next_state, reward, done, truncated, _ = env.step(action)
            
            gradients.append(grad)
            rewards.append(reward)

            state = next_state.reshape(1, -1)

        score = sum(rewards)
        scores.append(score)

        # Print episode status
        print(f"Episode: {episode} Score: {score}")

        # Monte-Carlo weight update step
        T = len(rewards)
        for t in range(T):
            # Calculate discounted return G_t from step t
            G_t = sum(gamma ** i * rewards[t + i] for i in range(T - t))
            # Weight update rule
            weight += alpha * G_t * gradients[t]

    return scores
