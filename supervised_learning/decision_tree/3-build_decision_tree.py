#!/usr/bin/env python3
"""
Module for building a decision tree - gathering leaves
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

    def count_nodes_below(self, only_leaves=False):
        """Counts the total number of nodes or leaves below this node"""
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
        leaves = []
        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves


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
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Returns itself inside a list"""
        return [self]


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
