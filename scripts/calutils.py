'''
Some utility functions
'''
import os
import numpy as np
import pandas as pd

# os.environ['AMPLIGRAPH_DATA_HOME'] = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), './ampligraph_datasets')


def normalise(x):
    '''

    '''
    return x / np.sum(x)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


def sigmoid(z):
    '''
    Sigmoid function

    Parameters
    ----------
    z: float - a real number
    
    Returns
    -------
    sig_val: float - a real number within (0, 1)
    ''' 
    sig_val = 1 / (1 + np.exp(-z))
    return sig_val
