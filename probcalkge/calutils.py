'''
Some utility functions
'''
import numpy as np
from scipy.special import expit

def get_cls_name(obj):
    '''Get the name of the class of the object
    
    Example
    -------
    >>> l = [1, 2, 3]
    >>> get_cls_name(l)
    'list'
    '''
    return obj.__class__.__name__

def get_func_name(func):
    '''Get the name of the given function
    
    Example
    >>> def hhh():pass
    >>> get_func_name(hhh)
    'hhh'
    '''
    return func.__name__

def sigmoid(z: float):
    '''
    Sigmoid function

    Parameters
    ----------
    z: float - a real number
    
    Returns
    -------
    sig_val: float - a real number within (0, 1)
    
    Example
    -------
    >>> sigmoid(6).round(3)
    0.998
    ''' 
    sig_val = 1 / (1 + np.exp(-z))
    return sig_val

def expit_probs(x: np.ndarray):
    '''
    convert scores into uncalibrated probabilities

    Parameters
    ----------
    x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    
    Returns
    -------
        uncalibrated probabilities

    Example
    -------
    >>> expit_probs(np.array([1, 3, 5])).round(3)
    array([0.731, 0.953, 0.993])
    '''
    return expit(x)


def normalise(x: np.ndarray):
    '''
    Normalise an array of number

    Parameters
    ----------
    x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    
    Returns
    -------
        normalised numbers within (0, 1)

    Example
    -------
    >>> normalise(np.array([1, 3, 5]))
    array([0.11111111, 0.33333333, 0.55555556])
    '''
    return x / np.sum(x)


def softmax(x: np.ndarray):
    '''
    Compute softmax values for each row of scores in x.
    
    Parameters
    ----------
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    
    Returns
    -------
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array

    Example
    -------
    >>> softmax(np.array([[1, 3, 5], [1, 4, 9]])).round(3)
    array([[0.016, 0.117, 0.867],
       [0.   , 0.007, 0.993]])
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


def oneD_to_twoD(array):
    '''convert 1D array into 2D array, else return the original array

    Example
    -------
    >>> oneD_to_twoD(np.array([1, 3, 5]))
    array([[1],
       [3],
       [5]])
    >>> oneD_to_twoD(np.array([[1, 3, 5]]))
    array([[1, 3, 5]])
    '''
    if len(array.shape) == 1: # 1D array
        array = array.reshape(-1, 1) # convert to 2D array
    return array