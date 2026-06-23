#!/usr/bin/env python3
"""
Module for building an Isolation Random Tree
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """Represents an Isolation Random Tree for outlier detection"""
    def __init__(self, max_depth=10, seed=0, root=None):
        """Initializes an Isolation Random Tree"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def depth(self):
        """Returns the maximum depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts total nodes or leaves in the entire tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """String representation of the entire tree"""
        return self.root.__str__()

    def get_leaves(self):
        """Returns a list of all leaves in the tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Triggers lower and upper bound tracking from root downwards"""
        self.root.update_bounds_below()

    def np_extrema(self, arr):
        """Returns the minimum and maximum of an array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Computes a random split criterion for a node"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Creates and returns a leaf child node"""
        leaf_child = Leaf(value=node.depth + 1, depth=node.depth + 1)
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates and returns an internal child node"""
        child = Node(depth=node.depth + 1)
        child.sub_population = sub_population
        return child

    def fit_node(self, node):
        """Recursively fits an isolation random tree node"""
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold
        )

        is_left_leaf = (
            node.depth + 1 >= self.max_depth
            or np.sum(left_population) <= 1
        )
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            node.depth + 1 >= self.max_depth
            or np.sum(right_population) <= 1
        )
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Fits the tree onto the explanatory data"""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(
            explanatory.shape[0], dtype='bool'
        )
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print(f"  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(
                f"    - Number of leaves          : "
                f"{self.count_nodes(only_leaves=True)}"
            )

    def update_predict(self):
        """Updates the prediction function to return the leaf depth"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            """Returns the depth of the leaf each individual falls into"""
            indicators = np.array([leaf.indicator(A) for leaf in leaves])
            leaf_indices = np.argmax(indicators, axis=0)
            return np.array([leaves[i].depth for i in leaf_indices])

        self.predict = predict_func
