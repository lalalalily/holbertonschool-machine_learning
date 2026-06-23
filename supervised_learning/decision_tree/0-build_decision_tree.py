#!/usr/bin/env python3
"""
Module for building a decision tree - depth
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
        if self.left_child is None and self.right_child is None:
            return self.depth
        left_d = 0
        right_d = 0
        if self.left_child is not None:
            left_d = self.left_child.max_depth_below()
        if self.right_child is not None:
            right_d = self.right_child.max_depth_below()
        return max(left_d, right_d)


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
