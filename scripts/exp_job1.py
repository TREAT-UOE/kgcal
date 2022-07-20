'''
A job script to be run on computing clusters
'''

import sys, os
from datetime import datetime
# enable importing the modules from probcalkge
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../'))
sys.path.append(os.path.join(ROOT_DIR, '../probcalkge'))

from probcalkge import Experiment
from probcalkge import get_calibrators
from probcalkge import get_datasets,  get_kgemodels
from probcalkge import brier_score, negative_log_loss, ks_error

time_str = datetime.now().strftime('%m-%d_%H-%M-%S')
SAVE_MODEL_PATH = os.path.join(ROOT_DIR, time_str)

ds = get_datasets()
cals = get_calibrators()
kges = get_kgemodels()

exp = Experiment(
    cals=[cals.uncal, cals.platt, cals.isot, cals.histbin, cals.beta], 
    # datasets=[ds.fb13, ds.wn18, ds.yago39, ds.dp50, ds.umls, ds.kinship, ds.nations], 
    datasets=[ds.fb13, ds.wn18, ds.yago39], 
    kges=[kges.transE, kges.complEx, kges.distMult, kges.hoLE, kges.convKB, kges.convE], 
    metrics=[brier_score, negative_log_loss, ks_error]
    )

exp.train_kges()
exp.save_trained_kges(SAVE_MODEL_PATH)
# exp.load_trained_kges(SAVE_MODEL_PATH)
exp_res = exp.run_with_trained_kges()
exp_res.report_html_file()

