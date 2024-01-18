import numpy as np
# Data analysis and manipulation tool for python
import pandas as pd

# Plotting library
import matplotlib.pyplot as plt 



__all__ = ['train_test_split']

def train_test_split(x, y, train_ratio=0.8):
    """
        Separate the dataset into training and testing dataset for learning and evaluating the model 
        of linear regression.
        
        Parameters
        ----------
        x : array_like, the input dataset of shape (m, n+1).        
        y : array_like, value at given features. A vector of shape (m, 1).
            
        train_size: float, the percetage of training dataset (between 0 and 1)
        
        Returns
        -------
        x_train : array_like, matrix of the training dataset.
        x_test : array_like, matrix of the testing dataset. 
        y_train : array_like, value at given features in training datset. A vector of shape (m, 1).
        y_test : array_like, value at given features in testing dataset. A vector of shape (m, 1).
    """
    
    m = x.shape[0]
    
    row_indices = np.random.permutation(m)
    
    training_set_num = int(train_ratio * m)
    
    
    # Create a Training Set
    x_train = x[row_indices[:training_set_num],:]
    y_train = y[row_indices[:training_set_num],:]

    # Create a Test Set
    x_test = x[row_indices[training_set_num:],:]
    y_test = y[row_indices[training_set_num:],:]
    
    return x_train, x_test, y_train, y_test