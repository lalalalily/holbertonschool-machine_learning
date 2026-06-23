#!/usr/bin/env python3
"""
Module for building an Isolation Random Forest
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """Represents an Isolation Random Forest"""
    def __init__(self, n_trees=100, max_depth=15, seed=0):
        """Initializes an Isolation Random Forest"""
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.numpy_predicts = []
        self.trees = []

    def fit(self, explanatory, verbose=0):
        """Fits the isolation forest to the training data"""
        for i in range(self.n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory, verbose=0)
            self.numpy_predicts.append(T.predict)
            self.trees.append(T)
        if verbose == 1:
            depths = [t.depth() for t in self.trees]
            nodes = [t.count_nodes() for t in self.trees]
            leaves = [t.count_nodes(only_leaves=True) for t in self.trees]
            print("Training finished.")
            print(f"    - Mean depth                     : {np.mean(depths)}")
            print(f"    - Mean number of nodes           : {np.mean(nodes)}")
            print(f"    - Mean number of leaves          : {np.mean(leaves)}")

    def suspects(self, explanatory, n_suspects=3):
        """Finds the anomalies/suspects with the minimum mean depth"""
        preds = np.array([pred_func(explanatory)
                          for pred_func in self.numpy_predicts])
        mean_depths = np.mean(preds, axis=0)
        sort_idx = np.argsort(mean_depths)
        suspect_indices = sort_idx[:n_suspects]
        return explanatory[suspect_indices], mean_depths[suspect_indices]
