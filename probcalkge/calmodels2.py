'''
Some calibration methods
'''
from abc import abstractmethod
from collections import namedtuple
from copy import deepcopy
from typing import Iterable

import numpy as np

from netcal.scaling import (
    TemperatureScaling,
    LogisticCalibration,
    BetaCalibration
)
from netcal.binning import (
    IsotonicRegression,
    HistogramBinning,
    BBQ,
    ENIR
)

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
        self._calibrator = LogisticCalibration()  

    def fit(self, uncal_probs, y):
        uncal_probs = oneD_to_twoD(uncal_probs)
        self._calibrator.fit(uncal_probs, y)
        return self

    def predict(self, uncal_probs):
        uncal_probs = oneD_to_twoD(uncal_probs)
        return self._calibrator.transform(uncal_probs)
    

class IsotonicCalibrator(Calibrator):
    def __init__(self):
        self.name = 'IsotonicCalibrator'
        self._calibrator = IsotonicRegression()

    def fit(self, uncal_probs, y):
        self._calibrator.fit(uncal_probs, y)
        return self
        
    def predict(self, uncal_probs):
        return self._calibrator.transform(uncal_probs)
    

class HistogramBinningCalibrator(Calibrator):
    """
    Histogram Binning as a calibration method. The bins are divided into equal lengths.
    """
    
    def __init__(self, M=15):
        """
        M (int): the number of equal-length bins used
        """
        self.name = 'HistogramBinningCalibrator'
        self._calibrator = HistogramBinning()  

    def fit(self, uncal_probs, true):
        self._calibrator.fit(uncal_probs, true)

    # Fit based on predicted confidence
    def predict(self, uncal_probs):
        return self._calibrator.transform(uncal_probs)
    

class BetaCalibrator(Calibrator):
    
    def __init__(self):
        self.name = 'BetaCalibrator'
        self._calibrator = BetaCalibration()
    
    def fit(self, uncal_probs, y):
        self._calibrator.fit(uncal_probs, y)
        return self
    
    def predict(self, uncal_probs):
        return self._calibrator.transform(uncal_probs)


class TemperatureCalibrator(Calibrator):

    def __init__(self) -> None:
        self.name = 'TemperatureCalibrator'
        self._calibrator = TemperatureScaling()
    
    def fit(self, uncal_probs, y):
        self._calibrator.fit(np.array(uncal_probs), np.array(y))
        return self

    def predict(self, uncal_probs):
        return self._calibrator.transform(np.array(uncal_probs))


class ENIRCalibrator(Calibrator):

    def __init__(self) -> None:
        self.name = 'ENIRCalibrator'
        self._calibrator = ENIR()
    
    def fit(self, uncal_probs, y):
        self._calibrator.fit(np.array(uncal_probs), np.array(y))
        return self

    def predict(self, uncal_probs):
        ps = self._calibrator.transform(np.array(uncal_probs))
        return ps

class BBQCalibrator(Calibrator):
    def __init__(self) -> None:
        self.name = 'BBQCalibrator'
        self._calibrator = BBQ()
    
    def fit(self, uncal_probs, y):
        self._calibrator.fit(np.array(uncal_probs), np.array(y))
        return self

    def predict(self, uncal_probs):
        ps = self._calibrator.transform(np.array(uncal_probs))
        return ps    


CalibraionModels = namedtuple('CalibrationModels', [
    'uncal', 
    'platt', 
    'isot', 
    'histbin', 
    'beta',
    'temperature',
    'enir',
    'bbq',
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
    lst.append(BBQCalibrator())
    return CalibraionModels(*lst)