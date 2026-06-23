#!/usr/bin/env python3
"""
Constructs unguided random partitioning trees designed for anomaly detection workflows.
"""
import numpy as np
Node = __import__('6-build_decision_tree').Node
Leaf = __import__('6-build_decision_tree').Leaf


class Isolation_Random_Tree:
    def __init__(self, max_depth=10, seed=0):
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)

    def fit_node(self, explanatory, depth=0):
        """Splits structural clusters recursively via randomly isolated feature intervals."""
        if depth >= self.max_depth or explanatory.shape[0] <= 1:
            leaf = Leaf(value=explanatory.shape[0], depth=depth)
            leaf.size = explanatory.shape[0]
            return leaf

        n_features = explanatory.shape[1]
        valid_features = [f for f in range(n_features) if len(np.unique(explanatory[:, f])) > 1]
        
        if not valid_features:
            leaf = Leaf(value=explanatory.shape[0], depth=depth)
            leaf.size = explanatory.shape[0]
            return leaf

        feature = self.rng.choice(valid_features)
        f_min, f_max = explanatory[:, feature].min(), explanatory[:, feature].max()
        threshold = self.rng.uniform(f_min, f_max)

        left_mask = explanatory[:, feature] > threshold
        right_mask = explanatory[:, feature] <= threshold

        node = Node(feature=feature, threshold=threshold, depth=depth)
        node.left_child = self.fit_node(explanatory[left_mask], depth + 1)
        node.right_child = self.fit_node(explanatory[right_mask], depth + 1)
        return node

    def fit(self, explanatory):
        """Initializes unguided tree fitting routines."""
        self.root = self.fit_node(explanatory)
