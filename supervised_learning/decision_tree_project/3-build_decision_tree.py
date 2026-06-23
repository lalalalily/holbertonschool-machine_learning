#!/usr/bin/env python3
"""
Adds a method to gather all leaf nodes from the decision tree.
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

    def left_child_add_prefix(self, text):
        """Adds formatting prefix lines for left children."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds formatting prefix lines for right children."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Returns the structured string representation of this subtree."""
        if self.is_root:
            out = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            out = f"node [feature={self.feature}, threshold={self.threshold}]\n"
        out += self.left_child_add_prefix(self.left_child.__str__())
        out += self.right_child_add_prefix(self.right_child.__str__())
        return out

    def get_leaves_below(self):
        """Recursively collects all leaves underneath this internal node."""
        return self.left_child.get_leaves_below() + self.right_child.get_leaves_below()


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

    def __str__(self):
        """Returns string representation for a leaf."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Returns a list containing only itself."""
        return [self]


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

    def __str__(self):
        """Returns the full tree layout string representation."""
        return self.root.__str__().strip() + "\n"

    def get_leaves(self):
        """Retrieves a list of all leaves across the tree."""
        return self.root.get_leaves_below()
