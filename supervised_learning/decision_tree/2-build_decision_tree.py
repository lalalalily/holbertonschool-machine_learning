#!/usr/bin/env python3
"""
Module for building and representing a Decision Tree.
"""
import numpy as np


class Node:
    """Represents an internal node in a decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes a Node instance."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculates the maximum depth of the tree below this node."""
        if self.left_child is None and self.right_child is None:
            return self.depth
        lengths = []
        if self.left_child:
            lengths.append(self.left_child.max_depth_below())
        if self.right_child:
            lengths.append(self.right_child.max_depth_below())
        return max(lengths) if lengths else self.depth

    def count_nodes_below(self, only_leaves=False):
        """Counts the number of nodes or leaves below this node."""
        if only_leaves:
            count = 0
            if self.left_child:
                count += self.left_child.count_nodes_below(only_leaves=True)
            if self.right_child:
                count += self.right_child.count_nodes_below(only_leaves=True)
            return count
        count = 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves=False)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves=False)
        return count

    def left_child_add_prefix(self, text):
        """Adds formatting prefixes for left children visualization."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds formatting prefixes for right children visualization."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        """Returns the string representation of the node."""
        if self.is_root:
            out = (f"root [feature={self.feature},"
                   f" threshold={self.threshold}]\n")
        else:
            out = (f"-> node [feature={self.feature},"
                   f" threshold={self.threshold}]\n")
        if self.left_child:
            out += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            out += self.right_child_add_prefix(str(self.right_child))
        return out


class Leaf(Node):
    """Represents a leaf node in a decision tree."""
    def __init__(self, value, depth=None):
        """Initializes a Leaf instance."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 since it is a leaf."""
        return 1

    def __str__(self):
        """Returns the string representation of the leaf."""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Represents a complete decision tree model."""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="gini", root=None):
        """Initializes a Decision Tree instance."""
        self.rng = np.random.default_rng(seed)
        self.root = root
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def depth(self):
        """Returns the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns the total number of nodes or leaves in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Returns the string representation of the entire tree."""
        return self.root.__str__().rstrip("\n") + "\n"
