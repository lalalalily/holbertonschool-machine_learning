#!/usr/bin/env python3
"""
Module for building a random forest classifier
"""
import numpy as np

Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """Represents a Random Forest classifier"""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize the Random Forest with given hyperparameters."""
        self.numpy_preds = []
        self.target = None
        self.explanatory = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Train the random forest by fitting n_trees decision trees."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []

        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )

            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)

            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}""")
            print("    - Accuracy of the forest on td   :"
                  f" {self.accuracy(self.explanatory, self.target)}")

    def predict(self, explanatory):
        """Predict class labels using majority vote across all trees."""
        preds = np.array([pred(explanatory) for pred in self.numpy_preds])

        return np.array([
            np.argmax(np.bincount(preds[:, i]))
            for i in range(preds.shape[1])
        ])

    def accuracy(self, explanatory, target):
        """Return the fraction of correct predictions on the given data."""
        return np.sum(self.predict(explanatory) == target) / target.size
