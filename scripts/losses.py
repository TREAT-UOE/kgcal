import sys
# enable importing the modules from probcalkge
sys.path.append('../')
sys.path.append('../probcalkge')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from probcalkge import Experiment, ExperimentResult
from probcalkge import get_calibrators
from probcalkge import get_datasets,  get_kgemodels
from probcalkge import brier_score, negative_log_loss, ks_error, ece

cals = get_calibrators()
ds = get_datasets()

for loss in ['nll', 'pairwise', 'self_adversarial', 'absolute_margin']:
    kges = get_kgemodels(loss)
    exp = Experiment(
        cals=[cals.uncal, cals.platt, cals.isot, ], 
        datasets=[ds.yago39], 
        kges=[kges.complEx, kges.distMult, kges.hoLE], 
        metrics=[ece]
        )

    exp.train_kges()
    exp.save_trained_kges(f'./saved_kges/{loss}/')
