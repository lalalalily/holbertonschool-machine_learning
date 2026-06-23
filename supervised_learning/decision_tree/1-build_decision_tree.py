#!/usr/bin/env python3
"""
Adds the ability to count all nodes or only leaves in the tree.
"""
import numpy as np


class Node:
    """Represents an internal node in a decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculates the maximum depth of nodes below this node."""
        return max(self.left_child.max_depth_below(), self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Counts the number of nodes or leaves below this node recursively."""
        if only_leaves:
            return self.left_child.count_nodes_below(only_leaves=True) + self.right_child.count_nodes_below(only_leaves=True)
        return 1 + self.left_child.count_nodes_below() + self.right_child.count_nodes_below()


class Leaf(Node):
    """Represents a leaf node in a decision tree."""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 for a leaf node."""
        return 1


class Decision_Tree():
    """Represents a decision tree classifier."""
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
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
        """Returns the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns the total count of nodes or leaves in the entire tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)
