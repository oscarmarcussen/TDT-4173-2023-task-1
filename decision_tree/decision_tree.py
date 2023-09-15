import numpy as np
import pandas as pd
from math import log2
from collections import Counter
import random


# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:

    def __init__(self, max_depth=4, min_samples_split=3, min_samples_leaf=1, min_info_gain=0.15):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.tree = {}
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_info_gain = min_info_gain

    def fit(self, X, y):
        """
        Generates a decision tree for classification

        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """

        """
        Building the tree using some hyperparameters
        """
        self.tree = self._build_tree(X, y)

    def _entropy(self, y):
        """
        Calculate the entropy of a set of labels y.
        """
        if len(y) == 0:
            return 0

        class_counts = Counter(y)
        entropy = 0
        total_samples = len(y)

        for count in class_counts.values():
            p = count / total_samples
            entropy -= p * log2(p)

        return entropy

    def _information_gain(self, X, y, feature):
        """
        Calculate the information gain for a specific feature.
        """
        total_entropy = self._entropy(y)
        values = X[feature].unique()
        weighted_entropy = 0

        for value in values:
            subset = y[X[feature] == value]
            weighted_entropy += (len(subset) / len(y)) * self._entropy(subset)

        information_gain = total_entropy - weighted_entropy
        return information_gain

    def _build_tree(self, X, y):
        """
        Recursively build the decision tree with hyperparameters.
        """
        # If all labels are the same or depth limit reached, return that label as a leaf node
        if len(set(y)) == 1 or self.max_depth == 0:
            return y.iloc[0]

        # If there are no features left to split on, return the most common label
        if len(X.columns) == 0:
            return y.mode()[0]

        # If the number of samples is less than the minimum for splitting, return a leaf node
        if len(X) < self.min_samples_split:
            return y.mode()[0]

        # Find the feature with the highest information gain
        max_info_gain = -1
        best_feature = None

        for feature in X.columns:
            info_gain = self._information_gain(X, y, feature)
            if info_gain > max_info_gain and info_gain >= self.min_info_gain:  # Check against min_info_gain
                max_info_gain = info_gain
                best_feature = feature

        # Check if the number of samples in the subset is less than the minimum for a leaf node
        if len(X) < self.min_samples_leaf:
            return y.mode()[0]

        # Create a sub-tree for each unique value of the best feature
        tree = {best_feature: {}}
        for value in X[best_feature].unique():
            # Split the data and labels based on the best feature's value
            X_subset = X[X[best_feature] == value].drop(columns=best_feature)
            y_subset = y[X[best_feature] == value]

            # Check if the number of samples in the subset is less than the minimum for a leaf node
            if len(X_subset) < self.min_samples_leaf:
                tree[best_feature][value] = y_subset.mode()[0]
            else:
                # Recursively build the sub-tree with reduced depth and hyperparameters
                tree[best_feature][value] = self._build_tree(X_subset, y_subset)

        return tree

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        if not self.tree:
            raise ValueError("The decision tree has not been trained. Call .fit() first.")

        predictions = []
        for _, row in X.iterrows():
            prediction = self._predict_tree(self.tree, row)
            predictions.append(prediction)

        return np.array(predictions)

    def _predict_tree(self, tree, sample):
        """
        Recursively navigate the decision tree to make predictions.
        """
        # If we reach a leaf node (a class label), return it as the prediction
        if not isinstance(tree, dict):
            return tree

        # Otherwise, find the feature in the current node and follow the branch
        feature = list(tree.keys())[0]
        value = sample[feature]

        if value in tree[feature]:
            # Recursively follow the branch
            return self._predict_tree(tree[feature][value], sample)
        else:
            # If the value is not in the tree, return the most common class label at this node
            return max(tree[feature], key=tree[feature].get)
        # raise NotImplementedError()
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        rules = []
        self._extract_rules(self.tree, [], rules)
        return rules

    def _extract_rules(self, node, antecedent, rules):
        if isinstance(node, dict):
            for feature, branches in node.items():
                for value, sub_node in branches.items():
                    new_antecedent = antecedent + [(feature, value)]
                    self._extract_rules(sub_node, new_antecedent, rules)
        else:
            label = node
            rules.append((antecedent, label))
        # raise NotImplementedError()


# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a length k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))



