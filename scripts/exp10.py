import os

from pykeen.pipeline import pipeline
from pykeen.datasets import PathDataset
from pykeen.optimizers import Adam
from pykeen.losses import SoftplusLoss
from pykeen.regularizers import LpRegularizer

DATA_PATH = r'/disk/scratch/s1904162/kgcal/data/inpupt/ampligraph_datasets'
SAVE_PATH = r'/disk/scratch/s1904162/kgcal/data/saved_models'

YAGO310 = PathDataset(
    training_path=os.path.join(DATA_PATH, 'YAGO3-10' + os.sep + 'train.txt'),
    testing_path=os.path.join(DATA_PATH, 'YAGO3-10' + os.sep + 'test.txt'),
    validation_path=os.path.join(DATA_PATH, 'YAGO3-10' + os.sep + 'valid.txt')
)
WN18RR = PathDataset(
    training_path=os.path.join(DATA_PATH, 'WN18RR' + os.sep + 'train.txt'),
    testing_path=os.path.join(DATA_PATH, 'WN18RR' + os.sep + 'test.txt'),
    validation_path=os.path.join(DATA_PATH, 'WN18RR' + os.sep + 'valid.txt')
)
FB15k237 = PathDataset(
    training_path=os.path.join(DATA_PATH, 'FB15K-237' + os.sep + 'train.txt'),
    testing_path=os.path.join(DATA_PATH, 'FB15K-237' + os.sep + 'test.txt'),
    validation_path=os.path.join(DATA_PATH, 'FB15K-237' + os.sep + 'valid.txt')
)


datasets = {
    'yago310': YAGO310,
    'wn18rr': WN18RR,
    'fb15k237': FB15k237
}


models = ['TransE', 'TransD', 'TransR', 'TransH', 'RotatE']

for dname, dataset in datasets.items():
    for model in models:
        pipeline_results = pipeline(
            dataset=dataset,
            model=model,
            loss=SoftplusLoss,
            optimizer=Adam,
            optimizer_kwargs=dict(lr=1e-4),
            stopper='early',
            stopper_kwargs=dict(frequency=10, patience=2, relative_delta=0.002),
            epochs=1,
        )
        pipeline_results.save_to_directory(os.path.join(SAVE_PATH, f'{model}-{dname}'))
