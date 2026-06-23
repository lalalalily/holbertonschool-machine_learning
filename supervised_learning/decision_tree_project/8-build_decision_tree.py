#!/usr/bin/env python3
"""
Upgrades classification nodes to pick best splits via Gini Impurity metric evaluation.
"""
import numpy as np
Node = __import__('7-build_decision_tree').Node
Leaf = __import__('7-build_decision_tree').Leaf
Decision_Tree = __import__('7-build_decision_tree').Decision_Tree


class Node(Node):
    def gini_impurity(self, labels):
        """Computes basic Gini structural cost score metric."""
        if len(labels) == 0:
            return 0
        counts = np.bincount(labels)
        probabilities = counts / len(labels)
        return 1 - np.sum(probabilities ** 2)

    def left_child_add_prefix(self, text):
        return "    +--" + text.replace("\n", "\n    |  ")
    def right_child_add_prefix(self, text):
        return "    +--" + text.replace("\n", "\n       ")

    def fit_node(self, target_criterion, current_population):
        """Recursive worker targeting minimum total Gini impurity costs."""
        if self.depth >= self.max_depth or len(current_population) <= self.min_pop or np.all(target_criterion == target_criterion[0]):
            return Leaf(value=np.bincount(target_criterion).argmax(), depth=self.depth)

        n_features = self.explanatory.shape[1]
        best_gini = 1.0
        best_feature = None
        best_threshold = None

        if self.split_criterion == "gini":
            for f in range(n_features):
                thresholds = np.unique(self.explanatory[current_population, f])
                for t in thresholds:
                    left_idx = self.explanatory[current_population, f] > t
                    right_idx = self.explanatory[current_population, f] <= t
                    
                    if not np.any(left_idx) or not np.any(right_idx):
                        continue
                        
                    gini_l = self.gini_impurity(target_criterion[left_idx])
                    gini_r = self.gini_impurity(target_criterion[right_idx])
                    
                    total_gini = (np.sum(left_idx) * gini_l + np.sum(right_idx) * gini_r) / len(current_population)
                    if total_gini < best_gini:
                        best_gini = total_gini
                        best_feature = f
                        best_threshold = t

            if best_feature is None:
                return Leaf(value=np.bincount(target_criterion).argmax(), depth=self.depth)

            self.feature = best_feature
            self.threshold = best_threshold

            left_mask = self.explanatory[current_population, self.feature] > self.threshold
            right_mask = self.explanatory[current_population, self.feature] <= self.threshold

            self.left_child = Node(depth=self.depth + 1)
            self.left_child.max_depth = self.max_depth
            self.left_child.min_pop = self.min_pop
            self.left_child.split_criterion = self.split_criterion
            self.left_child.explanatory = self.explanatory
            self.left_child = self.left_child.fit_node(target_criterion[left_mask], current_population[left_mask])

            self.right_child = Node(depth=self.depth + 1)
            self.right_child.max_depth = self.max_depth
            self.right_child.min_pop = self.min_pop
            self.right_child.split_criterion = self.split_criterion
            self.right_child.explanatory = self.explanatory
            self.right_child = self.right_child.fit_node(target_criterion[right_mask], current_population[right_mask])
            return self
        else:
            return super().fit_node(target_criterion, current_population)


class Decision_Tree(Decision_Tree):
    def fit(self, explanatory, target, verbose=0):
        self.explanatory = explanatory
        self.target = target
        self.root.max_depth = self.max_depth
        self.root.min_pop = self.min_pop
        self.root.split_criterion = self.split_criterion
        self.root.rng = self.rng
        self.root.explanatory = explanatory
        
        self.root = self.root.fit_node(target, np.arange(explanatory.shape[0]))
        self.update_predict()
