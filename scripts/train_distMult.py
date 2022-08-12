import sys
import os
# enable importing the modules from probcalkge
sys.path.append('../')
sys.path.append('../probcalkge')

from probcalkge import get_datasets

ds = get_datasets()

from ampligraph.latent_features import DistMult, HolE
from ampligraph.utils import save_model

SAVE_PATH = ''


hyperparams1 = {
    'verbose': True,
    'epochs': 200,

    # taken from https://arxiv.org/abs/1912.10000
    'k': 100,
    'optimizer': 'adam',
    'loss': 'nll',
    'eta': 20,
    'optimizer_params': {'lr': 1e-4},
}


hyperparams2 = {
    'verbose': True,
    'epochs': 300,

    # taken from https://arxiv.org/abs/1912.10000
    'k': 100,
    'optimizer': 'adam',
    'loss': 'nll',
    'eta': 20,
    'optimizer_params': {'lr': 1e-4},
}

hyperparams3 = {
    'verbose': True,
    'epochs': 200,
    'regularizer': 'LP', 
    'regularizer_params': {'p': 2},

    # taken from https://arxiv.org/abs/1912.10000
    'k': 100,
    'optimizer': 'adam',
    'loss': 'nll',
    'eta': 20,
    'optimizer_params': {'lr': 1e-4},
}

def run(hyperparams):
    for d in [ds.fb13, ds.wn18, ds.yago39]:
        m = DistMult(**hyperparams)
        m.fit(d.X_train)
        save_model(m, os.path.join(SAVE_PATH, f'{d.name}_DistMult_{hyperparams["epochs"]}.pkl'))

run(hyperparams1)
run(hyperparams2)
run(hyperparams3)
