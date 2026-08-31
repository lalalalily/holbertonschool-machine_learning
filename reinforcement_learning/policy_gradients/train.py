#!/usr/bin/env python3
"""Module containing policy gradient computation function."""
import numpy as np


def policy(state, weight):
    """Computes the softmax policy probabilities."""
    z = np.dot(state, weight)
    exp_z = np.exp(z - np.max(z))  # Stability shift
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """Computes the Monte-Carlo policy gradient for a given state and weight.

    Args:
        state: matrix of shape (1, n) or (n,) representing the environment state
        weight: matrix of shape (n, m) representing current policy weights

    Returns:
        action: the sampled action index
        grad: matrix of shape (n, m) representing the weight gradient
    """
    probs = policy(state, weight)
    
    # Sample action according to computed probabilities
    action = np.random.choice(len(probs[0]), p=probs[0])

    # Convert state to a 2D column vector (n, 1) if necessary
    state_vec = np.asarray(state).reshape(-1, 1)

    # One-hot representation of chosen action
    d_softmax = probs.copy()
    d_softmax[0, action] -= 1

    # Compute gradient: s^T * (probs - e_a)
    grad = np.dot(state_vec, d_softmax)

    return action, grad
