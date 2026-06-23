#!/usr/bin/env python3
"""
Defines Isolation Forest architectures mapping outlier paths globally.
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest:
    def __init__(self, n_trees=100, max_depth=10, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trees = []

    def fit(self, explanatory, verbose=0):
        """Trains ensemble isolation structures without target training vectors."""
        self.trees = []
        for i in range(self.n_trees):
            tree = Isolation_Random_Tree(max_depth=self.max_depth, seed=self.seed + i)
            tree.fit(explanatory)
            self.trees.append(tree)

    def path_length(self, x, node):
        """Computes the isolated depth traversal length score from structural subsets."""
        if node.is_leaf:
            if node.size <= 1:
                return node.depth
            # Returns internal estimation normalization scaling factors
            c = 2 * (np.log(node.size - 1) + 0.5772156649) - (2 * (node.size - 1) / node.size)
            return node.depth + c

        if x[node.feature] > node.threshold:
            return self.path_length(x, node.left_child)
        else:
            return self.path_length(x, node.right_child)

    def suspects(self, explanatory, n_suspects):
        """Extracts the topmost anomalous vectors matching critical depth scores."""
        # Calculate individual entry trace arrays across trees
        mean_depths = np.array([np.mean([self.path_length(row, tree.root) for tree in self.trees]) for row in explanatory])
        sorted_indices = np.argsort(mean_depths)
        return explanatory[sorted_indices[:n_suspects]], mean_depths[sorted_indices[:n_suspects]]
