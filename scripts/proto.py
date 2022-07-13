import random

import numpy as np
import tensorflow as tf
import pandas as pd

from scipy.special import expit
from ampligraph.latent_features import ComplEx, TransE, DistMult, HolE
from ampligraph.evaluation import generate_corruptions_for_fit, create_mappings, to_idx
from netcal.scaling import BetaCalibration 
from netcal.binning import IsotonicRegression, HistogramBinning
from netcal.metrics import ECE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
# %matplotlib inline


def negative_sampling(dat, eta=1):
    '''
    Given a set of positive triples, perform negative sampling by corruption
    return a set of (X, y)
    '''
    rel_to_idx, ent_to_idx = create_mappings(dat)
    dat_id = to_idx(dat, ent_to_idx, rel_to_idx)
    dat2 = generate_corruptions_for_fit(dat_id, eta=eta)

    dat2_id = dat2.eval(session=tf.compat.v1.Session())
    X = np.concatenate([dat_id, dat2_id])
    y = np.concatenate([np.ones(len(dat_id)), np.zeros(len(dat2_id))])
    idx_to_rel = {v:k for k,v in rel_to_idx.items()}
    idx_to_ent = {v:k for k,v in ent_to_idx.items()}
    C = []
    for x in X:
        h, r, t = x
        C.append([idx_to_ent[h], idx_to_rel[r], idx_to_ent[t]])

    return np.array(C), y

def filter_unseen(train, test, y_test):
    '''filter out triples from test whose entities do not appear in train'''
    entities = set()
    relations = set()
    for x in train:
        h, r, t = x
        entities.add(h)
        relations.add(r)
        entities.add(t)
    filtered_triple = []
    filtered_y = []
    for i in range(len(test)):
        h, r, t = test[i]
        if (h in entities) and (t in entities) and (r in relations):
            filtered_triple.append([h, r, t])
            filtered_y.append(y_test[i])
    return np.array(filtered_triple), np.array(filtered_y)


def cal_and_eval(model, calibrator, X_cal, y_cal, X_test, y_test):
    scores_cal = expit(model.predict(X_cal))    
    calibrator.fit(scores_cal, y_cal)
    scores_test = expit(model.predict(X_test))
    proba_test = calibrator.transform(scores_test)
    print('MES:', mean_squared_error(proba_test, y_test))
    ece = ECE()
    print('ECE:', ece.measure(proba_test, y_test))
    return proba_test

def make_synthetic_evidence(triples):
    '''make synthetic positive and negative evidence for triples'''
    evds = []
    for t in triples:
        num_evd = random.randint(0,100)
        evds.append(np.random.randint(2, size=num_evd))
    return evds

def filter_literals(triples):
    '''filter out triples containing literals'''
    dat = []
    for l in triples:
        if str(l[2]).startswith('<http'):
            dat.append(l)
    dat = np.array(dat)
    return dat


class Triple:
    def __init__(self, subj, pred, obj, mu=None, v=None):
        self.subj = subj
        self.pred = pred
        self.obj = obj
        self._beta_a = 0
        self._beta_b = 0
        if mu != None and v != None:
            self.assign_init_prob_mu_v(mu, v)
    
    def __str__(self):
        return f'{self.prob}::({self.subj}, {self.pred}, {self.obj})'
    
    def __repr__(self):
        return self.__str__()
    
    def _calculate_beta_mu_v(self, mu, v):
        '''calculate a and b of beta distribution by mean and sample size'''
        self._beta_a = mu * v
        self._beta_b = (1 - mu) * v
        return self._beta_a, self._beta_b
    
    @property
    def prob(self):
        return self._beta_a / (self._beta_a + self._beta_b)
    
    def assign_init_prob_mu_v(self, mu, v):
        self._calculate_beta_mu_v(mu, v)
    
    def update_beta_a_b(self, evds_stream):
        num_ones = list(evds_stream).count(1)
        num_zeros = list(evds_stream).count(0)
        self._beta_a += num_ones
        self._beta_b += num_zeros
        return self._beta_a, self._beta_b
    
    def to_list(self):
        return [self.subj, self.pred, self.obj, self.prob]

    def to_array(self):
        return np.array(self.to_list())
    
