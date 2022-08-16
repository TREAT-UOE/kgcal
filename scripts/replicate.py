import imp
from operator import ipow
import sys
import os

from ampligraph.latent_features import ComplEx, TransE
from ampligraph.utils import save_model, restore_model

from probcalkge import get_datasets, brier_score, negative_log_loss, ks_error, ece

ds = get_datasets()

SAVE_PATH = ''

losses = ['pairwise', 'nll', 'self_adversarial', 'multiclass_nll']


for loss in losses:
    hyperparams1 = {
        'verbose': True,

        # taken from https://arxiv.org/abs/1912.10000
        'k': 100,
        'optimizer': 'adam',
        'loss': loss,
        'eta': 20,
        'optimizer_params': {'lr': 1e-4},
        'epochs': 1000,
    }
    model = ComplEx(**hyperparams1)
    model.fit(ds.fb13.X_train, tensorboard_logs_path=)
    save_model(model, os.path.join(SAVE_PATH, f'Comp_{loss}.pkl'))

