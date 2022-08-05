'''
Knowledge Graph Embedding Models
'''

from collections import namedtuple
from tabnanny import verbose
from ampligraph.latent_features import (
    RandomBaseline,
    TransE, 
    ComplEx, 
    HolE, 
    DistMult,
    ConvE,
    ConvKB,
    )



KGEModels = namedtuple('KGEModels', [
    'random',
    'transE',
    'complEx',
    'distMult',
    'hoLE',
    'convKB',
    'convE',
])

def get_kgemodels() -> KGEModels:
    hyperparams = {
        'verbose': True,

        # taken from https://arxiv.org/abs/1912.10000
        'k': 100,
        'optimizer': 'adam',
        'loss': 'nll',
        'eta': 20,
        'optimizer_params': {'lr': 1e-4},
        # 'epochs': 1000,
    }
    lst = []
    lst.append(RandomBaseline())
    lst.append(TransE(**hyperparams))
    lst.append(ComplEx(**hyperparams))
    lst.append(DistMult(**hyperparams))
    lst.append(HolE(**hyperparams))
    lst.append(ConvKB(**hyperparams))
    lst.append(ConvE(verbose=True))
    return KGEModels(*lst)