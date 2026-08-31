#!/usr/bin/env python3
"""Full training with Monte-Carlo policy gradient (REINFORCE)."""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a full training.

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor

    Returns:
        all values of the score (sum of all rewards during one
        episode loop)
    """
    weight = np.random.rand(4, 2)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        score = 0

        while True:
            action, grad = policy_gradient(state, weight)
            next_state, reward, done, truncated, _ = env.step(action)

            grads.append(grad)
            rewards.append(reward)
            score += reward

            state = next_state

            if done or truncated:
                break

        for t, grad in enumerate(grads):
            future_rewards = sum(
                r * (gamma ** i) for i, r in enumerate(rewards[t:])
            )
            weight += alpha * grad * future_rewards

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores
