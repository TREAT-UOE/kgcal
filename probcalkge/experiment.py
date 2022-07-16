'''

'''
from collections import OrderedDict
from copy import deepcopy
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
    results for each metric of each calibrator of each dataset of each KGE
    '''
    def __init__(self, experiment: 'Experiment', results) -> None:
        '''
        Parameters
        ----------
        results: dict of dict of DataFrames
            Experiment results stored in a dict
        '''
        self.expriment = experiment
        self.results = self._create_xarray(results)

    def _create_xarray(self, results: dict) ->xr.DataArray:
        coords = {
            'cal': [get_cls_name(cal) for cal in self.expriment.cals], 
            'kge': [get_cls_name(kge) for kge in self.expriment.kges],
            'dataset': [ds.name for ds in self.expriment.datasets], 
            'metric': [metric.__name__ for metric in self.expriment.metrics]
        }
        data = []
        for kge_name, ds_dict in results.items():
            frames = []
            for ds_name, frame in ds_dict.items():
                frames.append(frame.to_numpy())
            data.append(frames)

        print(data)
        print(coords)
        
        return xr.DataArray(data=data, coords=coords)

    def print_summary(self):
        summary = f'''
        calibration techniques: {[get_cls_name(cal) for cal in self.expriment.cals]}
        KGE models: {[get_cls_name(kge) for kge in self.expriment.kges]}
        datasets: {[ds.name for ds in self.expriment.datasets]}
        metrics: {[get_cls_name(metric) for metric in self.expriment.metrics]}
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
        return self.results.sel(cal=cal, kge=kge).to_dataframe()



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


        self.trained_kge = {}
        for ds in self.datasets:
            self.trained_kge[ds.name] = {}
            for kge in self.kges:
                self.trained_kge[ds.name][get_cls_name(kge)] = None
        
        self.trained_cal = {}
        for ds in self.datasets:
            self.trained_cal[ds.name] = {}
            for kge in self.kges:
                self.trained_cal[ds.name][get_cls_name(kge)] = {}
                for cal in self.cals:
                    self.trained_cal[ds.name][get_cls_name(kge)][get_cls_name(cal)] = None
        

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

    def run(self) -> ExperimentResult:
        '''run this experiment and return experiment results
        '''
        res = {}
        for kge in self.kges:
            res[get_cls_name(kge)] = {}
            for ds in self.datasets:
                res[get_cls_name(kge)][ds.name] = \
                    self._train_and_eval(kge, ds)
        print(res)
        return ExperimentResult(experiment=self, results=res)

    def _train_and_eval(self, kge: EmbeddingModel, ds: DatasetWrapper) -> pd.DataFrame:
        '''
        train one one KGE model on one datasets, then
        train and evaluate all calibration models,
        and measure the performance using all metrics
        '''
        print(f'training {get_cls_name(kge)} on {ds.name} ...')

        # make a brand new (untrained) kge models
        new_kge = deepcopy(kge)

        new_kge.fit(ds.X_train)
        uncal_prob_valid = expit_probs(new_kge.predict(ds.X_valid))
        uncal_prob_test = expit_probs(new_kge.predict(ds.X_test))

        self.trained_kge[ds.name][get_cls_name(kge)] = new_kge

        cals_metrics = {}
        for cal in self.cals:
            # make a brand new (untrained) cal models
            new_cal = deepcopy(cal)
            new_cal.fit(uncal_prob_valid, ds.y_valid)
            cal_prob_test = new_cal.predict(uncal_prob_test)

            self.trained_cal[ds.name][get_cls_name(kge)][get_cls_name(cal)] = new_cal

            cells = {}
            for metric in self.metrics:
                cells[metric.__name__] = metric(ds.y_test, cal_prob_test)
            cals_metrics[get_cls_name(cal)] = pd.Series(cells)

        df = pd.DataFrame(data=cals_metrics)
        print(df)
        return df
            




