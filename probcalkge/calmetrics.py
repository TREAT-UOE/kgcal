'''
Some calibration metrics.
'''
from collections.abc import Iterable
import numpy as np

from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.calibration import CalibrationDisplay
from netcal.metrics import ECE

def brier_score(y_true, y_prob, sample_weight=None, pos_label=None) -> float:
    '''Brier score'''
    return brier_score_loss(y_true, y_prob, 
            sample_weight=sample_weight, pos_label=pos_label)

def negative_log_loss(y_true, y_prob, **kwargs) -> float:
    '''Negative Log Loss'''
    eps = kwargs.get('eps') if 'eps' in kwargs else 1e-7
    return log_loss(y_true, y_prob, eps=eps)

def ks_error(y_true, y_prob) -> float:
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

def ece(y_true, y_prob) -> float:
    '''Kolmogorov-Smirnov Calibration Error'''
    ece = ECE()
    return ece.measure(y_prob, y_true)

def cal_curve(y_true, y_prob, n_bins=5):
    '''Plot a calibration curve'''
    return CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=n_bins)

def accuracy(y_true, y_prob) -> float:
    '''classification accuracy ration (not a calibration metrics)'''
    pred = y_prob > 0.5
    return accuracy_score(y_true, pred)
    