#!/usr/bin/env python3
"""
Module for building a random forest classifier
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """Represents a Random Forest classifier"""
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initializes a Random Forest classifier"""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def fit(self, explanatory, target, n_trees_to_be_fitted=100, verbose=0):
        """Fits the random forest to the training data"""
        self.target = target
        for i in range(n_trees_to_be_fitted):
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop,
                              seed=self.seed + i, split_criterion="random")
            T.fit(explanatory, target, verbose=0)
            T.update_predict()
            self.numpy_predicts.append(T.predict)

    def predict(self, explanatory):
        """Predicts the classes for a given explanatory array"""
        preds = np.array([pred_func(explanatory)
                          for pred_func in self.numpy_predicts])
        modes = []
        for i in range(preds.shape[1]):
            counts = np.bincount(preds[:, i])
            modes.append(np.argmax(counts))
        return np.array(modes)

    def accuracy(self, explanatory, target):
        """Returns the accuracy of the random forest on a given dataset"""
        return np.sum(self.predict(explanatory) == target) / len(target)
