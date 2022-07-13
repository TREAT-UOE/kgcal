import os
import sys
import numpy as np
import pandas as np
from ampligraph.datasets import load_fb13, load_wn11, load_yago3_10
from ampligraph.latent_features import TransE, ComplEx, DistMult, HolE, ConvE, RandomBaseline
from ampligraph.evaluation import evaluate_performance, mrr_score
from ampligraph.utils import save_model

from calmetrics import brier_score, negative_log_loss, ks_error

MODELS_PATH = './saved_models/'

print('loading fb13 ...')
# fb13 = load_fb13()
# fb13 = load_wn11()
fb13 = load_wn11()

print('training TransE ...')
transE_model = ComplEx()
transE_model.fit(fb13['train'])

if not os.path.exists(MODELS_PATH + 'TransE.pkl'):
    save_model(transE_model, MODELS_PATH+'TransE.pkl')

positive_triples = fb13['train']

print('evaluating TransE ...')
# ranks = evaluate_performance(fb13['test'], transE_model, 
#             filter_triples=positive_triples, verbose=True)

# mrr = mrr_score(ranks)

pos_triples = fb13['valid'][fb13['valid_labels']]
neg_triples = fb13['valid'][~fb13['valid_labels']]
transE_model.calibrate(pos_triples, neg_triples)
probs = transE_model.predict_proba(fb13['test'])

bs = brier_score(fb13['test_labels'], probs)
nll = negative_log_loss(fb13['test_labels'], probs)
ks = ks_error(fb13['test_labels'], probs)

# print('TranE\n--------\n', 'mrr:', mrr, 'bs:', bs, 'nll:', nll, 'ks:', ks)
results = f'''
ComplEx
-------
BS:  {bs}
NLL: {nll}
KS:  {ks}

For wn11
'''

print(results)

with open('single_test_results_wn1.txt', 'a') as f:
    f.write(results)

