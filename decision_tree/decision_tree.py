import numpy as np 
import pandas as pd
import array
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        pass
    
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
        Calculating total entropy
        """
        tot_entropy_arr = y.value_counts().to_numpy(dtype=int)
        tot_entropy = entropy(tot_entropy_arr)
        print(tot_entropy)

        """
        Calculating entropy and information gain for each feature
        """

        data = X.merge(y.to_frame(), left_index=True, right_index=True)
        attr_names = X.columns.to_list()
        num_of_rows = X.shape[0]
        entropy_data = []
        info_gain_data = []

        for n in attr_names:
            value_count_df = X[n].value_counts()
            values = X[n].unique()
            information_number = 0 # Used for calculating information gain
            for v in values:
                filtered_data = data[data[n] == v]
                value_df = filtered_data[y.name].value_counts()
                entropy_arr = value_df.to_numpy(dtype=int)
                value_entropy = entropy(entropy_arr)
                quantity = value_count_df[v]/num_of_rows
                entropy_item = {'Attribute': n, 'Value': v, 'Quantity': quantity, 'Entropy': value_entropy}
                entropy_data.append(entropy_item)

                # Calculating information number
                information_number += quantity * value_entropy

            information_gain = tot_entropy - information_number
            info_gain_item = {'Attribute': n, 'Information_gain': information_gain}
            info_gain_data.append(info_gain_item)

        entropy_df = pd.DataFrame(entropy_data) # df with the entropy of the different values of the different attributes
        info_gain_df = pd.DataFrame(info_gain_data) # df with the information gain of each attributes
        print(entropy_df.head())

        # Sorting the df to get the attribute with the highest information gain
        sorted_info_gain_df = info_gain_df.sort_values(by='Information_gain', ascending=False)
        print(sorted_info_gain_df.head())

        max_info_feature = sorted_info_gain_df['Attribute'].iloc[0]
        print(max_info_feature)



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
        # TODO: Implement
        raise NotImplementedError()


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



