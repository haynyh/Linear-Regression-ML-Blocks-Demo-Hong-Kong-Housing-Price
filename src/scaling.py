import numpy as np
# Data analysis and manipulation tool for python
import pandas as pd

# Plotting library
import matplotlib.pyplot as plt 





__all__ = ['min_max_scaler', 'inverse_min_max_scaler', 'z_score_scaler', 'inverse_z_score_scaler', 'scale_feature', 'inverse_scale_feature']

def min_max_scaler(x, x_min=None, x_max=None):
    """
        feature scaling with min-max range normalization
        x : arrary_like
            dataset with several features
        x_min : float
            given maximal value of features. If this input are given, the dataset will be scaled according to this value.
            If not, this value will be calculated by the data themselves. 
        x_max : float
            given minimal value of features. If this input are given, the dataset will be scaled according to this value.
    """
    
    new_x = np.zeros_like(x) # create a new matrix "new_x" with the shape as the input matrix 'x'
    
    if x_min is None:
        x_min = np.min(x, axis=0)

    if x_max is None:
        x_max = np.max(x, axis=0)
    

    numerator = x - x_min
    denominator = x_max - x_min

    # Avoid division by zero
    new_x = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=new_x, where=denominator != 0)
    
    
    if x_min is None or x_max is None:
        x_min = None
        x_max = None
    x_range = x_max-x_min
    
    return new_x, x_min, x_max



def inverse_min_max_scaler(scaled_x, parameters):
    """
        inverse feature scaling of the min-max normalization
    """
    x_min, x_max = parameters
    return scaled_x * (x_max-x_min) + x_min



def z_score_scaler(x, x_mean=None, x_std=None):
    """
        feature scaling with Z-score normalization (standarlization)
    """
    
    new_x = np.zeros_like(x) # create a new matrix "new_x" with the shape as the input matrix 'x'


    if x_mean == None:
        x_mean = np.mean(x, axis=0)

    if x_std == None:
        x_std = np.std(x, axis=0)

    numerator = x - x_mean
    denominator = x_std
    
    new_x = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=new_x, where=denominator != 0)
    
    
    if x_mean is None or x_std is None: # Please do not change this line !!!
        x_mean = None
        x_std = None
    
    return new_x, x_mean, x_std



def inverse_z_score_scaler(scaled_x, parameters):
    """
        inverse feature scaling with Z-score normalization (standarlization)
    """
    x_mean, x_std = parameters
    return scaled_x * x_std + x_mean



def scale_feature(x_train, x_test, method='min_max'):
    """
        sacling the features in training and testing dataset 
        only with distribution of training dataset.
    """
    
    scaled_train_data = np.zeros_like(x_train)
    scaled_test_data = np.zeros_like(x_test)
    
    if method == 'min_max':
        scaled_train_data, train_x_min, train_x_max = min_max_scaler(x_train)
        scaled_test_data, train_x_min, train_x_max = min_max_scaler(x_test, train_x_min, train_x_max)
        parameters = (train_x_min, train_x_max)
    elif method == 'z_score':
        scaled_train_data, train_x_mean, train_x_std = z_score_scaler(x_train)
        scaled_test_data, train_x_mean, train_x_std = z_score_scaler(x_test, train_x_mean, train_x_std)
        parameters = (train_x_mean, train_x_std)
    else:
        raise ValueError("The mentioned method have not been implemented yet, \
                         please select one from min-max and z-score normalization")
    
    return scaled_train_data, scaled_test_data, parameters


def inverse_scale_feature(scaled_x, parameters, method='min_max'):
    """
        inverse sacling the features in training and testing dataset 
        only with distribution of training dataset.
    """
    
    data_x = np.zeros_like(scaled_x)
        
    if method == 'min_max':
        data_x = inverse_min_max_scaler(scaled_x, parameters)
    elif method == 'z_score':
        data_x = inverse_z_score_scaler(scaled_x, parameters)
    else:
        raise ValueError("The mentioned method have not been implemented yet, \
                        please select one from min-max and z-score normalization")
    
    return data_x