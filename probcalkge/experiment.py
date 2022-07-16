'''

'''
import os
from collections import OrderedDict
from copy import deepcopy
from pprint import pprint
from typing import Callable, Iterable

import pandas as pd
import xarray as xr
from ampligraph.latent_features import EmbeddingModel

from calmodels import Calibrator
from caldatasets import DatasetWrapper
from calutils import expit_probs, get_cls_name

dict = OrderedDict

class ExperimentResult:
    '''
    results for each metric of each calibrator of each KGE of each dataset
    '''
    def __init__(self, experiment: 'Experiment', results) -> None:
        '''
        Parameters
        ----------
        results: dict of dict of DataFrames
            Experiment results stored in a dict[dataset, dict[kge, dataframe]]
            column and row(index) of the dataframe is cal and metric 
        '''
        self.expriment = experiment
        self.results = self._create_xarray(results)

    def _create_xarray(self, results: dict) ->xr.DataArray:
        print(results)

        dims = ['dataset', 'kge', 'cal', 'metric']
        coords = {
            'dataset': [ds.name for ds in self.expriment.datasets],
            'kge': [get_cls_name(kge) for kge in self.expriment.kges],
            'cal': [get_cls_name(cal) for cal in self.expriment.cals], 
            'metric': [metric.__name__ for metric in self.expriment.metrics]
        }
        data = []
        for kge_name, ds_dict in results.items():
            frames = []
            for ds_name, frame in ds_dict.items():
                frames.append(frame.to_numpy().T)
            data.append(frames)

        print(data)
        print(coords)
        
        return xr.DataArray(data=data, coords=coords, dims=dims, name='ExpRes')

    def print_summary(self):
        summary = f'''
        calibration techniques: {[get_cls_name(cal) for cal in self.expriment.cals]}
        KGE models: {[get_cls_name(kge) for kge in self.expriment.kges]}
        datasets: {[ds.name for ds in self.expriment.datasets]}
        metrics: {[metric.__name__ for metric in self.expriment.metrics]}
        '''
        print(summary)


    def print_full_report(self):
        '''Print the full report of an experiment
        '''
        self.print_summary()
        print(self.to_frame())

    def to_frame(self) -> pd.DataFrame:
        '''Return the result as a multi-indexed DataFrame
        '''
        return self.results.to_dataframe()
    
    def report_html_file(self, filename='report.html'):
        html_content = self.to_frame().to_html()
        with open(filename, 'w') as f:
            f.write(html_content)

    def slice(self, cal: str, kge: str) -> pd.DataFrame:
        '''Return a 2-D dataframe given a calibration model and a KGE model

        Parameters
        ----------
        cal - name of calibration model
        kge - name of KGE model

        Returns
        -------
            a DataFrame containing metrics of the given calibration model
            for a KGE model on various datasets
        '''
        return self.results.sel(cal=cal, kge=kge)\
                        .to_dataframe()\
                        .pivot_table(columns='dataset', index='metric', values='ExpRes')



# MetricFunction - takes a list of true labels and a list of probabilities, 
# output a score to indicate how well the probabilities 
# are calibrated. The scores should be the lower the better.
# For esample: brier_score(y_true, y_prob) -> float
MetricFunction = Callable[[Iterable[int], Iterable[float]], float]


class Experiment:
    '''
    Automated experiment to examine performance of calibration
    techqnieus for KGE models on different datasets
    '''
    def __init__(self, cals: Iterable[Calibrator],
                        kges: Iterable[EmbeddingModel],
                        datasets: Iterable[DatasetWrapper],
                        metrics: Iterable[MetricFunction]):
        self.cals = list(cals)
        self.kges = list(kges)
        self.datasets = list(datasets)
        self.metrics = list(metrics)

        self.trained_kges = {}
        for ds in self.datasets:
            self.trained_kges[ds.name] = {}
            for kge in self.kges:
                self.trained_kges[ds.name][get_cls_name(kge)] = None
        
        self.trained_cals = {}
        for ds in self.datasets:
            self.trained_cals[ds.name] = {}
            for kge in self.kges:
                self.trained_cals[ds.name][get_cls_name(kge)] = {}
                for cal in self.cals:
                    self.trained_cals[ds.name][get_cls_name(kge)][get_cls_name(cal)] = None
        

    def add_calmodel(self, calmodel: Calibrator):
        '''add an calibration model to be examined in this experiment
        '''
        self.cals.append(calmodel)

    def add_kgemodel(self, kgemodel: EmbeddingModel):
        '''add an KGE model in this experiment
        '''
        self.kges.append(kgemodel)

    def add_dataset(self, dataset: DatasetWrapper):
        '''add a dataset in this experiment
        '''
        self.datasets.append(dataset)

    def add_metric(self, metric: MetricFunction):
        '''add a metric function in this experiment
        '''
        self.metrics.append(MetricFunction)

    def save_trained_kges(self, save_dir: str):
        '''
        Save trained KGE models to a given directory,
        model names follow such a format '{dataset}-{KGE name}.pkl',
        such as 'FB13k-TransE.pkl' 
        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('made model directory:', save_dir)

        from ampligraph.utils import save_model
        for ds, models in self.trained_kges.items():
            for mname, model in models.items():
                model_file = os.path.join(save_dir, f'{ds}-{mname}.pkl')
                save_model(model, model_file)
                print(f'saved {model_file}.')

    def load_trained_kges(self, save_dir: str):
        '''
        Load trained KGE models from a given directory,

        Requires
        --------
        Model files should be produced by the `self.save_trained_kge` method
        '''
        from ampligraph.utils import restore_model
        import re
        fname_pat = re.compile('(.*?)-(.*?)\.pkl')
        for fname in os.listdir(save_dir):
            if fname_pat.match(fname):
                dsname, kgename =  fname_pat.findall(fname)[0]
                self.trained_kges.setdefault(dsname, dict())
                self.trained_kges[dsname][kgename] = restore_model(os.path.join(save_dir, fname))
        print('Loaded models:')
        pprint(self.trained_kges)

        
    def train_kges(self):
        '''train all KGE models on all datasets
        '''
        for kge in self.kges:
            for ds in self.datasets:
                trained_kge = self._train_kge(kge, ds)
                self.trained_kges[ds.name][get_cls_name(kge)] = trained_kge

    def _train_kge(self, kge: EmbeddingModel, ds: DatasetWrapper) -> EmbeddingModel:
        '''train one KGE model on one dataset
        '''
        print(f'training {get_cls_name(kge)} on {ds.name} ...')
        new_kge = deepcopy(kge)  # make a brand new (untrained) kge models
        new_kge.fit(ds.X_train)
        new_kge._is_trained = True
        return new_kge

    def _train_cal_and_eval(self, trained_kge: EmbeddingModel, 
                                ds: DatasetWrapper) -> pd.DataFrame:
        '''
        train and evaluate all calibration models for one KGE on one dataset,
        and measure the performance using all metrics
        '''
        # assert(trained_kge._is_trained, 'KGE model not trained!')

        uncal_prob_valid = expit_probs(trained_kge.predict(ds.X_valid))
        uncal_prob_test = expit_probs(trained_kge.predict(ds.X_test))

        cals_metrics = {}
        for cal in self.cals:
            new_cal = deepcopy(cal) # make a brand new (untrained) cal models
            new_cal.fit(uncal_prob_valid, ds.y_valid)
            cal_prob_test = new_cal.predict(uncal_prob_test)

            self.trained_cals[ds.name][get_cls_name(trained_kge)][get_cls_name(cal)] = new_cal

            cells = {}
            for metric in self.metrics:
                cells[metric.__name__] = metric(ds.y_test, cal_prob_test)
            cals_metrics[get_cls_name(cal)] = pd.Series(cells)

        df = pd.DataFrame(data=cals_metrics)
        # print(df)
        return df
            

    def run_with_trained_kges(self) -> ExperimentResult:
        '''run this experiments with all trained KGE models on all datasets
        '''
        res = {}
        for ds in self.datasets:
            res[ds.name] = {}
            for kgename, kgemodel in self.trained_kges[ds.name].items():
                res[ds.name][kgename] = self._train_cal_and_eval(kgemodel, ds)
        print(res)
        return ExperimentResult(experiment=self, results=res)
        
    def run(self) -> ExperimentResult:
        '''run this experiment
        '''
        self.train_kges()
        return self.run_with_trained_kges()





if __name__ == '__main__':
    exp = Experiment([],[],[],[])
    exp.load_trained_kges('../saved_models/07-16_15-18-09')
    print(exp.trained_kge)



