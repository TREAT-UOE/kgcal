'''
Probability Calibration for KG Embedding in Entity typing task
======

Test various calibration techniques for TransE, ComplEx, HoLE, 
DistMult on two datasets includeing entity type information: 
[YAGO-ET and DBpedia-ET](https://github.com/JunhengH/joie-kdd19/tree/master/data).

We check if 
1. the calibration techqniues work;
2. using calibrated probability and a natural threshold 0.5 could achieve STOA as fine-tuned scores
'''

import sys
# enable importing the modules from probcalkge
sys.path.append('../')
sys.path.append('../probcalkge')

from typing import Iterable, Callable, Union
import numpy as np
import pandas as pd

from probcalkge import Experiment, ExperimentResult
from probcalkge import get_calibrators
from probcalkge import get_datasets,  get_kgemodels
from probcalkge import brier_score, negative_log_loss, ks_error, ece

ds = get_datasets()
# from probcalkge.calmodels import get_calibrators
from probcalkge.calmodels2 import get_calibrators
cals = get_calibrators()
kges = get_kgemodels()

exp = Experiment(
    cals=[cals.uncal, cals.platt, cals.isot, cals.histbin, cals.beta], 
    datasets=[ds.yago_et, ds.dbpedia_et], 
    kges=[kges.transE, kges.complEx, kges.distMult, kges.hoLE], 
    metrics=[brier_score, negative_log_loss, ks_error, ece]
    )

exp.train_kges()
exp.save_trained_kges('../saved_kges')
exp_res = exp.run_with_trained_kges()

exp_res.to_frame().to_csv('res.csv')



