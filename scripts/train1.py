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

from ampligraph.latent_features import ComplEx
from ampligraph.utils import save_model

SAVE_PATH = ''

hyperparams1 = {
    'verbose': True,

    # taken from https://arxiv.org/abs/1912.10000
    'k': 100,
    'optimizer': 'adam',
    'loss': 'absolute_margin',
    'eta': 20,
    'optimizer_params': {'lr': 1e-4},
    'epochs': 300,
}

model1 = ComplEx(**hyperparams1)
model1.fit(ds.fb13.X_train)
save_model(model1, SAVE_PATH+'ComplEx_fb13_mar.pkl')

model2 = ComplEx(**hyperparams1)
model2.fit(ds.wn18.X_train)
save_model(model2, SAVE_PATH+'ComplEx_wn11_mar.pkl')

model3 = ComplEx(**hyperparams1)
model3.fit(ds.yago39.X_train)
save_model(model3, SAVE_PATH+'ComplEx_yg39_mar.pkl')

hyperparams2 = {
    'verbose': True,
    'regularizer': 'LP', 
    'regularizer_params': {'p': 3, 'lambda':0.1},

    # taken from https://arxiv.org/abs/1912.10000
    'k': 100,
    'optimizer': 'adam',
    'loss': 'absolute_margin',
    'eta': 20,
    'optimizer_params': {'lr': 1e-4},
    'epochs': 300,
}

model4 = ComplEx(**hyperparams2)
model4.fit(ds.fb13.X_train)
save_model(model4, SAVE_PATH+'ComplEx_fb13_mar_reg.pkl')

model5 = ComplEx(**hyperparams2)
model5.fit(ds.wn18.X_train)
save_model(model5, SAVE_PATH+'ComplEx_wn11_mar_reg.pkl')

model6 = ComplEx(**hyperparams2)
model6.fit(ds.yago39.X_train)
save_model(model6, SAVE_PATH+'ComplEx_yg39_mar_reg.pkl')