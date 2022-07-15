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
    DistMult
    )



KGEModels = namedtuple('KGEModels', [
    'random',
    'transE',
    'complEx',
    'distMult',
    'hoLE',
])

def get_kgemodels() -> KGEModels:
    lst = []
    lst.append(RandomBaseline())
    lst.append(TransE(verbose=True))
    lst.append(ComplEx(verbose=True))
    lst.append(DistMult(verbose=True))
    lst.append(HolE(verbose=True))
    return KGEModels(*lst)