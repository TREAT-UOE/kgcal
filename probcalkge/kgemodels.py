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
    lst = []
    lst.append(RandomBaseline())
    lst.append(TransE(verbose=True))
    lst.append(ComplEx(verbose=True))
    lst.append(DistMult(verbose=True))
    lst.append(HolE(verbose=True))
    lst.append(ConvKB(verbose=True))
    lst.append(ConvE(verbose=True))
    return KGEModels(*lst)