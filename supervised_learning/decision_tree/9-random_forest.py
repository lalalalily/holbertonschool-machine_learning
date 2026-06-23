#!/usr/bin/env python3
"""
Defines the Random_Forest ensemble classification architecture wrapper class.
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.numpy_rng = np.random.default_rng(seed)
        self.numpy_predict = None
        self.trees = []

    def fit(self, explanatory, target, verbose=0):
        """Fits multi-estimator nodes leveraging bootstrapping methods iteratively."""
        self.trees = []
        for i in range(self.n_trees):
            # Create bootstrapped sample indices
            indices = self.numpy_rng.choice(explanatory.shape[0], size=explanatory.shape[0], replace=True)
            tree = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop, seed=self.seed + i, split_criterion="gini")
            tree.fit(explanatory[indices], target[indices])
            self.trees.append(tree)

    def predict(self, explanatory):
        """Gathers majority class predictions across all estimators safely."""
        tree_preds = np.array([tree.predict(explanatory) for tree in self.trees])
        # Find absolute mode values element-wise across column vectors
        return np.array([np.bincount(tree_preds[:, col]).argmax() for col in range(explanatory.shape[0])])
