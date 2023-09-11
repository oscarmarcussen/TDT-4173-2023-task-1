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
        # TODO: Implement

        """
        Calculating total entropy
        """
        tot_entropy_arr = y.value_counts().to_numpy(dtype=int)
        tot_entropy = entropy(tot_entropy_arr)
        print(tot_entropy)

        """
        Calculating entropy for each feature
        """

        data = X.merge(y.to_frame(), left_index=True, right_index=True)
        attr_names = X.columns.to_list()
        items_data = []

        for n in attr_names:
            values = X[n].unique()
            for v in values:
                filtered_data = data[data[n] == v]
                entropy_arr = filtered_data[y.name].value_counts().to_numpy(dtype=int)
                value_entropy = entropy(entropy_arr)
                item = {'Label': n, 'Value': v, 'Entropy': value_entropy}
                items_data.append(item)

        entropy_df = pd.DataFrame(items_data) # df with the entropy of the different values of the different attributes
        print(entropy_df.head())

        """
        Calculating entropy for each feature
        
        feature_value_data: df containing data with a specific value of a feature (eg. Outlook = Sunny)
        label: string, name of the label of the df (=PlayTennis)
        class_list: list, unique classes of the label (=[Yes, No])
        
        returns float, calculated entropy of the feature value df (eg. for Outlook = Sunny returns 0.971)
        """
        """
        def calc_entropy(feature_value_data, label, class_list):
            class_count = feature_value_data.shape[0]
            entropy = 0

            for c in class_list:
                label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
                entropy_class = 0
                if label_class_count != 0:
                    probability_class = label_class_count/class_count #probability of the class
                    entropy_class = - probability_class * np.log2(probability_class) #entropy
                entropy += entropy_class
            return entropy

        df = X.merge(y.to_frame(), left_index=True, right_index=True)

        print(df.head())
        """


        #raise NotImplementedError()

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
        # TODO: Implement 
        raise NotImplementedError()
    
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



