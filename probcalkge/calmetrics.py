'''
Some calibration metrics.
'''
from collections.abc import Iterable
import numpy as np

from sklearn.metrics import brier_score_loss, log_loss

def brier_score(y_true, y_prob, sample_weight=None, pos_label=None):
    '''Brier score'''
    return brier_score_loss(y_true, y_prob, 
            sample_weight=sample_weight, pos_label=pos_label)

def negative_log_loss(y_true, y_prob, **kwargs):
    '''Negative Log Loss'''
    eps = kwargs.get('eps') if 'eps' in kwargs else 1e-7
    return log_loss(y_true, y_prob, eps=eps)

def ks_error(y_true, y_prob):
    '''Kolmogorov-Smirnov Calibration Error'''
    order = np.argsort(y_prob)
    probs = y_prob[order]
    labels = y_true[order]

    # the largest difference between cumulative distrbution of predicted probs
    # and cumulative distrbution of actual probs
    N = len(y_true)
    cumulative_forecast = np.cumsum(probs) / N
    cumulative_actual   = np.cumsum(labels) / N
    KS_error_max = np.amax(np.absolute(cumulative_forecast - cumulative_actual))
    return KS_error_max

