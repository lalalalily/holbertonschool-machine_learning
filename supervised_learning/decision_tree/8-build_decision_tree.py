#!/usr/bin/env python3
"""
Module for building a decision tree - Gini implementation
"""
import numpy as np


class Node:
    """Represents a node in a decision tree"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes a node"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculates the maximum depth below the current node"""
        if self.is_leaf:
            return self.depth
        left_d = 0
        right_d = 0
        if self.left_child is not None:
            left_d = self.left_child.max_depth_below()
        if self.right_child is not None:
            right_d = self.right_child.max_depth_below()
        return max(left_d, right_d)

    def count_nodes_below(self, only_leaves=False):
        """Counts the total number of nodes or leaves below this node"""
        if self.is_leaf:
            return 1
        if only_leaves:
            left_c = 0
            right_c = 0
            if self.left_child is not None:
                left_c = self.left_child.count_nodes_below(only_leaves)
            if self.right_child is not None:
                right_c = self.right_child.count_nodes_below(only_leaves)
            return left_c + right_c
        left_c = 0
        right_c = 0
        if self.left_child is not None:
            left_c = self.left_child.count_nodes_below(only_leaves)
        if self.right_child is not None:
            right_c = self.right_child.count_nodes_below(only_leaves)
        return 1 + left_c + right_c

    def left_child_add_prefix(self, text):
        """Adds prefix for left child string representation"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds prefix for right child string representation"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """String representation of a node"""
        if self.is_root:
            out = f"root [feature={self.feature}, " \
                  f"threshold={self.threshold}]\n"
        else:
            out = f"node [feature={self.feature}, " \
                  f"threshold={self.threshold}]\n"

        if self.left_child is not None:
            out += self.left_child_add_prefix(str(self.left_child))
        if self.right_child is not None:
            out += self.right_child_add_prefix(str(self.right_child))
        return out

    def get_leaves_below(self):
        """Returns a list of all leaves below this node"""
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Recursively computes the lower and upper bounds for each node"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        if self.is_leaf:
            return

        if self.left_child is not None:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child is not None:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """Computes the indicator function from lower and upper bounds"""
        def is_large_enough(x):
            """Checks if features are > lower bounds"""
            return np.all(np.array([
                x[:, key] > self.lower[key] for key in self.lower
            ]), axis=0)

        def is_small_enough(x):
            """Checks if features are <= upper bounds"""
            return np.all(np.array([
                x[:, key] <= self.upper[key] for key in self.upper
            ]), axis=0)

        self.indicator = lambda x: np.all(np.array([
            is_large_enough(x), is_small_enough(x)
        ]), axis=0)

    def pred(self, x):
        """Predict value for a single individual recursively"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Represents a leaf node in a decision tree"""
    def __init__(self, value, depth=None):
        """Initializes a leaf node"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf node"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 since it's a leaf node"""
        return 1

    def __str__(self):
        """String representation of a leaf"""
        return f"-> leaf [value={self.value}]\n"

    def get_leaves_below(self):
        """Returns itself inside a list"""
        return [self]

    def update_bounds_below(self):
        """Leaf bounds updates do nothing"""
        pass

    def pred(self, x):
        """Predict value for a single individual"""
        return self.value


class Decision_Tree():
    """Represents a decision tree"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes a decision tree"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

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

    def update_predict(self):
        """Updates the prediction function for the tree"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            """Predicts classes for an array of individuals"""
            indicators = np.array([leaf.indicator(A) for leaf in leaves])
            leaf_indices = np.argmax(indicators, axis=0)
            return np.array([leaves[i].value for i in leaf_indices])

        self.predict = predict_func

    def pred(self, x):
        """Predict value for a single individual using the tree"""
        return self.root.pred(x)

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

    def possible_thresholds(self, node, feature):
        """Returns midpoints between consecutive unique feature values"""
        values = np.unique(
            (self.explanatory[:, feature])[node.sub_population]
        )
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Finds best threshold for one feature using Gini impurity.

        Vectorized computation over all thresholds and classes at once.
        No for or while loops used.
        Returns (best_threshold, best_gini_split_value).
        """
        thresholds = self.possible_thresholds(node, feature)
        if len(thresholds) == 0:
            return (None, np.inf)

        # Feature values and targets for this node's sub-population
        feat_vals = self.explanatory[:, feature][node.sub_population]
        labels = self.target[node.sub_population]
        n = len(labels)

        classes = np.unique(labels)

        # Shape: (n_individuals, n_thresholds)
        # True where individual's feature > threshold => goes LEFT
        goes_left = feat_vals[:, np.newaxis] > thresholds[np.newaxis, :]

        # Shape: (n_individuals, n_classes)
        # True where individual belongs to class k
        is_class = labels[:, np.newaxis] == classes[np.newaxis, :]

        # Shape: (n_individuals, n_thresholds, n_classes)
        # Left_F[i, j, k] = True iff individual i goes left at threshold j
        #                    AND individual i is of class k
        left_f = goes_left[:, :, np.newaxis] & is_class[:, np.newaxis, :]

        # Count per (threshold, class) for left and right children
        # Shape: (n_thresholds, n_classes)
        left_counts = left_f.sum(axis=0).astype(float)
        right_counts = (is_class[:, np.newaxis, :] & ~goes_left[
            :, :, np.newaxis]).sum(axis=0).astype(float)

        # Total per threshold: shape (n_thresholds,)
        n_left = left_counts.sum(axis=1)
        n_right = right_counts.sum(axis=1)

        # Avoid division by zero
        n_left_safe = np.where(n_left == 0, 1, n_left)
        n_right_safe = np.where(n_right == 0, 1, n_right)

        # Gini impurity = 1 - sum(p_k^2)
        # Shape: (n_thresholds,)
        gini_left = 1 - np.sum(
            (left_counts / n_left_safe[:, np.newaxis]) ** 2, axis=1
        )
        gini_right = 1 - np.sum(
            (right_counts / n_right_safe[:, np.newaxis]) ** 2, axis=1
        )

        # Weighted average Gini split
        gini_split = (n_left * gini_left + n_right * gini_right) / n

        # Mask out invalid splits (empty left or right)
        valid = (n_left > 0) & (n_right > 0)
        gini_split = np.where(valid, gini_split, np.inf)

        best_idx = np.argmin(gini_split)
        return thresholds[best_idx], gini_split[best_idx]

    def Gini_split_criterion(self, node):
        """Finds best feature and threshold using Gini impurity"""
        X = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])
        ])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    def fit_node(self, node):
        """Recursively fits a node in the decision tree"""
        node_target = self.target[node.sub_population]

        is_pure = np.all(node_target == node_target[0])
        too_small = np.sum(node.sub_population) < self.min_pop
        max_depth_reached = node.depth == self.max_depth

        if too_small or max_depth_reached or is_pure:
            if len(node_target) == 0:
                val = 0
            else:
                counts = np.bincount(node_target)
                val = np.argmax(counts)
            leaf = Leaf(value=val, depth=node.depth)
            leaf.sub_population = node.sub_population
            # Replace this node in its parent by updating in-place
            node.is_leaf = True
            node.value = val
            node.left_child = None
            node.right_child = None
            return

        if self.split_criterion == "random":
            feature, threshold = self.random_split_criterion(node)
        else:
            feature, threshold = self.Gini_split_criterion(node)

        node.feature = feature
        node.threshold = threshold

        left_pop = node.sub_population & (
            self.explanatory[:, feature] > threshold
        )
        right_pop = node.sub_population & (
            self.explanatory[:, feature] <= threshold
        )

        if np.sum(left_pop) == 0 or np.sum(right_pop) == 0:
            counts = np.bincount(node_target)
            node.is_leaf = True
            node.value = np.argmax(counts)
            node.left_child = None
            node.right_child = None
            return

        node.left_child = Node(depth=node.depth + 1)
        node.left_child.sub_population = left_pop
        self.fit_node(node.left_child)

        node.right_child = Node(depth=node.depth + 1)
        node.right_child.sub_population = right_pop
        self.fit_node(node.right_child)

    def fit(self, explanatory, target, verbose=0):
        """Fits the decision tree to the training data"""
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones(target.shape[0], dtype=bool)
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print("  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(
                f"    - Number of leaves          : "
                f"{self.count_nodes(only_leaves=True)}"
            )
            print(
                f"    - Accuracy on training data : "
                f"{self.accuracy(explanatory, target)}"
            )

    def accuracy(self, explanatory, target):
        """Returns the accuracy of the model on the given dataset"""
        return np.sum(self.predict(explanatory) == target) / len(target)
