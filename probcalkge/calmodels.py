'''
Some calibration methods
'''
from abc import abstractmethod
from collections import namedtuple
from copy import deepcopy
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration
from netcal.scaling import TemperatureScaling
from netcal.binning import ENIR

from calutils import oneD_to_twoD
        

# ===============================================================
#                   Calibration Models
# ===============================================================


# from typing import Protocol
# class Calibrator(Protocol):
class Calibrator:
    '''
    Base class for all probability calibration models
    '''
    
    name: str

    @abstractmethod
    def fit(self, uncal_probs: Iterable[Iterable[float]], truth: Iterable[int]):
        """
        Fit the calibration model
        
        Params:
            uncal_probs: uncalibrated probabilities of data (shape [samples, type_of_probs])
            truth: true labels of data
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, uncal_probs):
        """
        Calibrate the probabilities
        
        Param:
            uncal_probs: probabilities of the data (shape [samples, type_of_probs])
            
        Returns:
            well-alibrated probabilities (shape [samples, classes])
        """
        raise NotImplemented
    

class UncalCalibrator(Calibrator):
    def __init__(self):
        self.name = 'UncalCalibrator'       

    def fit(self, uncal_probs, y):
        pass
    
    def predict(self, uncal_probs):
        return uncal_probs

class PlattCalibrator(Calibrator):
    def __init__(self):
        self.name = 'PlattCalibrator'
        self._calibrator = LogisticRegression()  

    def fit(self, uncal_probs, y):
        uncal_probs = oneD_to_twoD(uncal_probs)
        self._calibrator.fit(uncal_probs, y)
        return self

    def predict(self, uncal_probs):
        uncal_probs = oneD_to_twoD(uncal_probs)
        return self._calibrator.predict_proba(uncal_probs)[:, 1]
    
class IsotonicCalibrator(Calibrator):
    def __init__(self):
        self.name = 'IsotonicCalibrator'
        self._calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    def fit(self, uncal_probs, y):
        self._calibrator.fit(uncal_probs, y)
        return self
        
    def predict(self, uncal_probs):
        return self._calibrator.predict(uncal_probs)
    

class HistogramBinningCalibrator(Calibrator):
    """
    Histogram Binning as a calibration method. The bins are divided into equal lengths.
    """
    
    def __init__(self, M=15):
        """
        M (int): the number of equal-length bins used
        """
        self.name = 'HistogramBinningCalibrator'
        self.bin_size = 1./M  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1+self.bin_size, self.bin_size)  # Set bin bounds for intervals
 
    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        """
        Inner method to calculate optimal confidence for certain probability range
        
        Params:
            - conf_thresh_lower (float): start of the interval (not included)
            - conf_thresh_upper (float): end of the interval (included)
            - probs : list of probabilities.
            - true : list with true labels, where 1 is positive class and 0 is negative).
        """

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered)/nr_elems  # Sums positive classes
            return conf  

    def fit(self, uncal_probs, true):
        conf = []

        # Got through intervals and add confidence to list
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs = uncal_probs, true = true)
            conf.append(temp_conf)

        self.conf = conf

    # Fit based on predicted confidence
    def predict(self, uncal_probs):
        # Go through all the probs and check what confidence is suitable for it.
        probs = deepcopy(uncal_probs)
        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob)
            probs[i] = self.conf[idx]

        return probs
    

class BetaCalibrator(Calibrator):
    
    def __init__(self):
        self.name = 'BetaCalibrator'
        self._calibrator = BetaCalibration()
    
    def fit(self, uncal_probs, y):
        self._calibrator.fit(uncal_probs, y)
        return self
    
    def predict(self, uncal_probs):
        return self._calibrator.predict(uncal_probs)


class TemperatureCalibrator(Calibrator):

    def __init__(self) -> None:
        self.name = 'TemperatureCalibrator'
        self._calibrator = TemperatureScaling()
    
    def fit(self, uncal_probs, y):
        self._calibrator.fit(np.array(uncal_probs), np.array(y))
        return self

    def predict(self, uncal_probs):
        return self._calibrator.predict(np.array(uncal_probs))


class ENIRCalibrator(Calibrator):

    def __init__(self) -> None:
        self.name = 'ENIRCalibrator'
        self._calibrator = ENIR()
    
    def fit(self, uncal_probs, y):
        self._calibrator.fit(np.array(uncal_probs), np.array(y))
        return self

    def predict(self, uncal_probs):
        return self._calibrator.predict(np.array(uncal_probs))


CalibraionModels = namedtuple('CalibrationModels', [
    'uncal', 
    'platt', 
    'isot', 
    'histbin', 
    'beta',
    'temperature',
    'enir',
])

def get_calibrators() -> CalibraionModels:
    lst = []
    lst.append(UncalCalibrator())
    lst.append(PlattCalibrator())
    lst.append(IsotonicCalibrator())
    lst.append(HistogramBinningCalibrator())
    lst.append(BetaCalibrator())
    lst.append(TemperatureCalibrator())
    lst.append(ENIRCalibrator())
    return CalibraionModels(*lst)