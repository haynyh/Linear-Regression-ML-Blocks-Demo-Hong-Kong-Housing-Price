
import numpy as np
# Data analysis and manipulation tool for python
import pandas as pd

# Plotting library
import matplotlib.pyplot as plt 




__all__ = ['hypothesis', 'cost_computation', 'regularized_cost_computation', 'gradient_descent', 'regularized_gradient_descent', 'evaluation']


def hypothesis(theta, x):
    """
        Hypothesis function with linear model. 
        with parameters theta for linear regression and data points in x.
        
        Parameters
        ----------
        theta: array_like
            The parameters for the regression function. This is a vector of 
            shape (n+1, 1).
        
        x : array_like
            The input dataset of shape (m, n+1), where m is the number of examples,
            and n is the number of features. Assume that a vector of one's already 
            appended to the features so the n+1 columns are given.
        
        Returns
        -------
        h : array_like
            Predicted values at given features. A vector of shape (m, 1).
    """
    
    h = np.zeros((x.shape[0],))
    
    # multiplication between matrix x and vector theta (mxn) * (nxm)
    h = np.matmul(x, theta)
    
    return h



def cost_computation(theta, x, y):
    """
    Cost function for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in x and y.
    
    Parameters
    ----------
    theta : array_like
        The parameters for the regression function. This is a vector of 
        shape (n+1, 1).
        
    x : array_like
        The input dataset of shape (m, n+1), where m is the number of examples,
        and n is the number of features. Assume a vector of one's already 
        appended to the features so the n+1 columns are given.
    
    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, 1).
        
    Returns
    -------
    cost : float
        The value of cost function.
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta and return it. 
    """
    
    m = x.shape[0] #num of x
    cost = .0
    
    # 1. compute the hypothesis value
    # 2. compute the error between hypothesis and y with np.substract
    # 3. compute the squared error (np.power)
    # 4. compute the cost value (np.sum)
    hyp = hypothesis(theta, x)
    errors = np.subtract(hyp, y)
    squared_errors = np.power(errors,2)
    cost = np.sum(squared_errors) / (2*m)
    
    return cost



def regularized_cost_computation(theta, x, y, lamda):
    """
    Cost function for linear regression with a regularization term. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in x and y.
    
    Parameters
    ----------
    theta : array_like
        The parameters for the regression function. This is a vector of 
        shape (n+1, 1).
        
    x : array_like
        The input dataset of shape (m, n+1), where m is the number of examples,
        and n is the number of features. Assume that a vector of one's already 
        appended to the features so n+1 columns are given.
    
    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, 1).
        
    lamda : float
        Hyperparameter for regularization term.
        
    Returns
    -------
    cost : float
        The value of cost function.
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta and return it. 
    """
    
    m = x.shape[0]
    cost = .0
    

    # 1. compute the hypothesis value
    # 2. compute the error between hypothesis and y with np.substract "errors"
    # 3. compute the squared error "squared_errors" (np.power)
    # 4. compute the cost value "error_cost" (np.sum)
    # 5. compute the regularization cost value "regularization_cost"
    hyp = hypothesis(theta, x)
    errors = np.subtract(hyp, y)
    squared_errors = np.power(errors,2)
    error_cost = np.sum(squared_errors) / (2*m)
    #regularization_cost = (lamda / (2 * m)) * np.dot(theta[1:].T, theta[1:]).item()
    regularization_cost = (lamda / (2 * m)) * np.dot(theta.T,theta).item()

    cost = error_cost + regularization_cost
    
    return cost





def gradient_descent(theta, x, y, alpha):
    """
    Performs gradient descent to learn `theta`. Updates theta with only one iteration,
    i.e., one gradient step with learning rate `alpha`.
    
    Parameters
    ----------
    theta : array_like
        Initial values for the linear regression parameters. 
        A vector of shape (n+1, 1).
        
    x : array_like
        The input dataset of shape (m, n+1).
    
    y : array_like
        Value at given features. A vector of shape (m, 1).
    
    alpha : float
        The learning rate.
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, 1).
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.
    """
    
    # Initialize some useful values
    m = y.shape[0] 
    n = theta.shape[0]
    new_theta = np.zeros((n, 1))
    
    hyp = hypothesis(theta, x)
    hyp_diff = np.subtract(hyp, y)
    for j in range(n):
        x_column = np.reshape(x[:, j], (-1, 1)) # make sure this is a 2D array with shape (m, 1)
        error_list = np.multiply(hyp_diff, x_column)
        total_error = np.sum(error_list)
        new_theta[j] = theta[j] - (alpha/m)*total_error
    
    return new_theta





def regularized_gradient_descent(theta, x, y, alpha, lamda):
    """
    Performs gradient descent with regulariztion to learn `theta`. Updates theta with only one iteration,
    i.e., one gradient step with learning rate `alpha`.
    
    Parameters
    ----------
    theta : array_like
        Initial values for the linear regression parameters. 
        A vector of shape (n+1, 1).
        
    x : array_like
        The input dataset of shape (m, n+1).
    
    y : array_like
        Value at given features. A vector of shape (m, 1).
    
    alpha : float
        The learning rate.
        
    lamda : float
        hyperparameter for regularization term.
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, 1).
    
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.
    """
    
    m = x.shape[0]
    n = theta.shape[0]
    new_theta = np.zeros((n, 1))
    

    hyp = hypothesis(theta, x)
    hyp_diff = np.subtract(hyp, y)
    for j in range(n):
        x_column = np.reshape(x[:, j], (-1, 1)) # make sure this is a 2D array with shape (m, 1)
        error_list = np.multiply(hyp_diff, x_column)
        total_error = np.sum(error_list)
        new_theta[j] = theta[j]*(1-alpha*lamda/m) - (alpha/m)*total_error

    
    return new_theta



def evaluation(theta, x, y):
    """
        evaluates the sum of squares due to error.
        
        Parameters
        ----------
        theta : array_like
            Initial values for the linear regression parameters. 
            A vector of shape (n+1, 1).
            
        x : array_like
            The input dataset of shape (m, n+1).
        
        y : array_like
            Value at given features. A vector of shape (m, 1).
        
        Returns
        -------
        mse : float
            the sum of squares due to error 
    """
    
    mse = .0
    m = x.shape[0]

    hyp = hypothesis(theta,x)
    hyp_diff = np.subtract(hyp, y)
    squared_errors = np.power(hyp_diff,2)
    mse = np.sum(squared_errors)/m

    
    return mse