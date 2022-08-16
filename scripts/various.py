import sys
import os
sys.path.append('../')
sys.path.append('../probcalkge')

from ampligraph.latent_features import ComplEx, DistMult, HolE
from ampligraph.utils import save_model, restore_model

from probcalkge import get_datasets, brier_score, negative_log_loss, ks_error, ece, get_cls_name

ds = get_datasets()

SAVE_PATH = '/disk/scratch/s1904162/kgcal/data/saved_models/'

losses = ['pairwise', 'nll', 'self_adversarial', 'multiclass_nll']
epoches = [1, 100, 300]
Models = [ComplEx, DistMult, HolE]

for data in [ds.fb13, ds.wn18, ds.yago39]:
    for loss in losses:
        for epoch in epoches:
            for Model in Models:
                hyperparams1 = {
                    'verbose': True,

                    'k': 100,
                    'optimizer': 'adam',
                    'loss': loss,
                    'eta': 20,
                    'optimizer_params': {'lr': 1e-4},
                    'epochs': epoch,
                }
                model = Model(**hyperparams1)
                model.fit(data.X_train)
                save_model(model, os.path.join(SAVE_PATH, f'{data.name}_{loss}_{get_cls_name(Model)}_{epoch}.pkl'))