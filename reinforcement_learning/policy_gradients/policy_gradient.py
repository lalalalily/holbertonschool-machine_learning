def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state
    and a weight matrix.

    Args:
        state: matrix representing the current observation of
            the environment
        weight: matrix of random weight

    Returns:
        the action and the gradient (in this order)
    """
    state = state.reshape(1, -1)
    probs = policy(state, weight)

    action = np.random.choice(probs.shape[1], p=probs[0])

    s = probs.reshape(-1, 1)
    softmax_derivative = np.diagflat(s) - np.dot(s, s.T)

    dlog = softmax_derivative[action] / probs[0, action]

    gradient = state.T.dot(dlog.reshape(1, -1))

    return action, gradient
