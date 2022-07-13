'''
Some utility functions
'''
import numpy as np



def sigmoid(z: float):
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


def normalise(x: np.ndarray):
    '''
    Normalise an array of number

    Parameters
    ----------
    x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    
    Returns
    -------
        a real number within (0, 1)
    '''
    return x / np.sum(x)


def softmax(x: np.ndarray):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters
    ----------
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    
    Returns
    -------
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


def oneD_to_twoD(array):
    '''convert 1D array into 2D array
    '''
    if len(array.shape) == 1: # 1D array
        array = array.reshape(-1, 1) # convert to 2D array
    return array