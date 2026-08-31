#!/usr/bin/env python3
"""Module that contains the train function for Monte-Carlo policy gradient."""
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Implements a full Monte-Carlo policy gradient training loop.

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: learning rate
        gamma: discount factor
        show_result: boolean, default False, render env every 1000 episodes

    Returns:
        scores: list of all values of the score per episode
    """
    weight = np.random.rand(4, 2)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        state = state.reshape(1, -1)

        gradients = []
        rewards = []

        render_this_episode = show_result and (episode % 1000 == 0)

        done = False
        truncated = False

        while not (done or truncated):
            if render_this_episode:
                env.render()

            action, grad = policy_gradient(state, weight)
            next_state, reward, done, truncated, _ = env.step(action)

            gradients.append(grad)
            rewards.append(reward)

            state = next_state.reshape(1, -1)

        if render_this_episode:
            env.render()

        score = sum(rewards)
        scores.append(score)

        print(f"Episode: {episode} Score: {score}")

        # Compute discounted returns and update weights using policy gradients
        T = len(rewards)
        for t in range(T):
            G_t = sum(gamma ** i * rewards[t + i] for i in range(T - t))
            weight += alpha * G_t * gradients[t]

    return scores
