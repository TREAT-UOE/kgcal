
'''
Utilities for loeading datasets
'''
import os
from collections import namedtuple

import numpy as np
from ampligraph.datasets import load_fb13, load_wn11, load_yago3_10, load_cn15k, load_nl27k


class DatasetWrapper:
    '''Adapter to wrap a dataset in order to fit our experiments
    '''
    def __init__(self, name, X_train, X_valid, y_valid, X_test, y_test):
        self.name = name
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test


def get_fb13() -> DatasetWrapper:
    tmp = load_fb13()
    return DatasetWrapper('FB13k', tmp['train'], 
                          tmp['valid'], 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'], 
                          tmp['test_labels'].astype(np.int32))

def get_wn11() -> DatasetWrapper:
    tmp = load_wn11()
    return DatasetWrapper('WN11', tmp['train'], 
                          tmp['valid'], 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'], 
                          tmp['test_labels'].astype(np.int32))

def _load_yago39():
    yago_path = os.environ['AMPLIGRAPH_DATA_HOME'] + os.sep + 'yago39' + os.sep
    data = {}
    with open(yago_path + 'train_triple2id.txt', 'r') as f:
        lines = f.readlines()
        data['train'] = np.array([line.strip().split() for line in lines[1:]])
    train_entities = set(data['train'][:, 0]).union(set(data['train'][:, 2]))
    print(len(train_entities))

    with open(yago_path + 'valid_triple2id_positive.txt', 'r') as f:
        lines = f.readlines()
        tmp = []
        for line in lines[1:]:
            triple = line.strip().split()
            if (triple[0] in train_entities) and (triple[2] in train_entities):
                tmp.append(triple)
        data['valid'] = np.array(tmp)
        data['valid_labels'] = np.ones(len(tmp))

    with open(yago_path + 'valid_triple2id_negative.txt', 'r') as f:
        lines = f.readlines()
        tmp = []
        for line in lines[1:]:
            triple = line.strip().split()
            if (triple[0] in train_entities) and (triple[2] in train_entities):
                tmp.append(triple)
        data['valid'] = np.concatenate([data['valid'], np.array(tmp)])
        data['valid_labels'] = np.concatenate([data['valid_labels'], np.zeros(len(tmp))])

    with open(yago_path + 'test_triple2id_positive.txt', 'r') as f:
        lines = f.readlines()
        tmp = []
        for line in lines[1:]:
            triple = line.strip().split()
            if (triple[0] in train_entities) and (triple[2] in train_entities):
                tmp.append(triple)
        data['test'] = np.array(tmp)
        data['test_labels'] = np.ones(len(tmp))

    with open(yago_path + 'test_triple2id_negative.txt', 'r') as f:
        lines = f.readlines()
        tmp = []
        for line in lines[1:]:
            triple = line.strip().split()
            if (triple[0] in train_entities) and (triple[2] in train_entities):
                tmp.append(triple)
        data['test'] = np.concatenate([data['test'], np.array(tmp)])
        data['test_labels'] = np.concatenate([data['test_labels'], np.zeros(len(tmp))])
    valid_entities = set(data['valid'][:, 0]).union(set(data['valid'][:, 2]))
    test_entities = set(data['test'][:, 0]).union(set(data['test'][:, 2]))
    print(len(valid_entities - train_entities))
    print(len(test_entities - train_entities))

    return data


def get_yago39() -> DatasetWrapper:
    tmp = _load_yago39()
    return DatasetWrapper('YAGO39', tmp['train'].astype(np.int32), 
                          tmp['valid'].astype(np.int32), 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'].astype(np.int32), 
                          tmp['test_labels'].astype(np.int32))


ExperimentDatasets = namedtuple('Datasets', [
    'fb13', 
    'wn18', 
    'yago39',
])

def get_datasets() -> ExperimentDatasets:
    lst = []
    lst.append(get_fb13())
    lst.append(get_wn11())
    lst.append(get_yago39())
    return ExperimentDatasets(*lst)