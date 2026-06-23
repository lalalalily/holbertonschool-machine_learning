#!/usr/bin/env python3
"""
Implements the recursive node splitting mechanism ('fit') under random selection rules.
"""
import numpy as np
# Import previous tasks structural requirements safely
Node = __import__('6-build_decision_tree').Node
Leaf = __import__('6-build_decision_tree').Leaf
Decision_Tree = __import__('6-build_decision_tree').Decision_Tree


def np_split(array, feature, threshold):
    """Splits array indices safely depending on criteria boundaries."""
    return array[:, feature] > threshold, array[:, feature] <= threshold


class Node(Node):
    def missing_child_policy(self, current_criterion, current_population):
        """Handles leaf fallbacks when missing target samples."""
        pass

    def fit_node(self, target_criterion, current_population):
        """Recursively matches feature boundaries to split population arrays."""
        # Check termination criteria
        if self.depth >= self.max_depth or len(current_population) <= self.min_pop or np.all(target_criterion == target_criterion[0]):
            node_value = np.bincount(target_criterion).argmax()
            return Leaf(value=node_value, depth=self.depth)

        if self.split_criterion == "random":
            # Select feature randomly
            possible_features = [f for f in range(self.explanatory.shape[1]) if len(np.unique(self.explanatory[current_population, f])) > 1]
            if not possible_features:
                node_value = np.bincount(target_criterion).argmax()
                return Leaf(value=node_value, depth=self.depth)

            self.feature = self.rng.choice(possible_features)
            self.threshold = self.rng.choice(np.unique(self.explanatory[current_population, self.feature]))

            # Split populations
            left_mask, right_mask = np_split(self.explanatory[current_population], self.feature, self.threshold)
            left_pop = current_population[left_mask]
            right_pop = current_population[right_mask]

            if len(left_pop) == 0 or len(right_pop) == 0:
                node_value = np.bincount(target_criterion).argmax()
                return Leaf(value=node_value, depth=self.depth)

            # Assign recursive children objects safely
            self.left_child = Node(depth=self.depth + 1)
            self.left_child.max_depth = self.max_depth
            self.left_child.min_pop = self.min_pop
            self.left_child.split_criterion = self.split_criterion
            self.left_child.rng = self.rng
            self.left_child.explanatory = self.explanatory
            self.left_child = self.left_child.fit_node(target_criterion[left_mask], left_pop)

            self.right_child = Node(depth=self.depth + 1)
            self.right_child.max_depth = self.max_depth
            self.right_child.min_pop = self.min_pop
            self.right_child.split_criterion = self.split_criterion
            self.right_child.rng = self.rng
            self.right_child.explanatory = self.explanatory
            self.right_child = self.right_child.fit_node(target_criterion[right_mask], right_pop)

            return self


class Decision_Tree(Decision_Tree):
    def fit(self, explanatory, target, verbose=0):
        """Fits the training datasets systematically into nodes."""
        self.explanatory = explanatory
        self.target = target
        self.root.max_depth = self.max_depth
        self.root.min_pop = self.min_pop
        self.root.split_criterion = self.split_criterion
        self.root.rng = self.rng
        self.root.explanatory = explanatory
        
        initial_pop = np.arange(explanatory.shape[0])
        self.root = self.root.fit_node(target, initial_pop)
        self.update_predict()
