'''

'''
from typing import Iterable

import pandas as pd
from ampligraph.latent_features import EmbeddingModel

from calmodels import Calibrator, DatasetWrapper



class Experiment:
    '''
    Automated experiment to examine performance of calibration
    techqnieus for KGE models on different datasets
    '''
    def __init__(self):
        pass

    def add_calmodel(self, calmodel: Calibrator):
        '''add an calibration model to be examined in this experiment
        '''
        pass

    def add_kgemodel(self, kgemodel: EmbeddingModel):
        '''add an KGE model in this experiment
        '''
        pass

    def add_dataset(self, dataset: DatasetWrapper):
        '''add a dataset in this experiment
        '''
        pass


    def run_exp(self):
        '''run this experiment and print results
        '''
        pass

