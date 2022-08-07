# REvisiting Probability Calibration in Knowledge Graph Embedding

## Introduction



1. We reprecated and extended the previous experiments. We confirmed their conclusion but also pointed out the potential flaws of previous works:
    - Calibration techqniues like Platt Scaling, Isotonic Regression can effectively convert embedding scores into good probabilities.
    - However, converting embedding scores via expit transform (simply pss the scores through a sigmoid layer) is not a good idea, because the transformed probabilities are not only uncalibrated, but also can't be used by calibration. Arguably, some may even refuse to recognise these transformed values as probabilities. A better idea is to convert the embedding scores to probabilities directly via platt scaling or isontonic regression, instead of expit transform.
2. We also found a more useful rule of thumb about how to choose calibration techqniue: when we have a large set of held-out data (over 10 thousand triples), we should definitely choose binning-based calibration techniques, such as Isonotic Regression, otherwise scaling-based techqniues, such as Platt Scaling, would be more suitable. 


## From Embedding Scores to Probabilities


## Results

In our experiment, we train 4 typical KE embedding models, TransE, ComplEx, DistMult, and HoLE on 7 datasets: FB13k, WN11, YAGO39, DBpedia50, Nations, Kinship, UMLS. Each dataset is splited into 3 subset for train, valid, and test. Note that the valid and test set of FB13, WN11 and YAGO39 have ground truth negative samples, while the other 4 don't have. Thus, we generate synthetic negative samples via the corrption and local closed world assumtion. Statistics of the datasets are summarisd in table 1.

|           | train  | valid | test  |
|-----------|--------|-------|-------|
| FB13k     | 316232 | 11816 | 47464 |
| WN11      | 110361 | 4877  | 19706 |
| YAGO39    | 354994 | 18474 | 18514 |
| DBpedia50 | 32388  | 246   | 4196  |
| UMLS      | 5216   | 1304  | 1322  |
| Kinship   | 8544   | 2136  | 2148  |
| Nations   | 1592   | 398   | 402   |

We run calibration techniques for every KG Embedding models. During experiment, we train the KGE models only use the train set, and train the calibration model using valid set. In diagram 1, the calibration curves illustrate that all calibration techqniues produced better-calibrated probabilities than those obtained via expit transform.

![diagram 1](curves2.png)

We can see that binning-based calibration performs better in general. We also noticed that binning-based methods dominate in FB13k, WN11 and YAGO39. We hypothesise that perhaps that's because there are more data in these 3 datasets. Previous work also suggested that binnig-based methods tend to overfit, expecially in smaller datasets. Thus, we take these 3 datasets, and gradaully shrink the size of valid set by randomly samping k%, and compare the number of winning between binning-based and scaling-based methods. 

![Diagram 2](shrink2.png)

Diagram 2 shows that as the size of the dataset shrink, the winning count of binning-based methods decreases while caling-based method increases, i.e., scaling-based calibration start to gain better calibration peroformance than binning-based techqniue in some cases.

| FB13k     | Uncal | Platt | Isot  | beta  | histbin |
|-----------|-------|-------|-------|-------|---------|
| TransE    | 0.500 | 0.672 | 0.672 | 0.673 | 0.668   |
| ComplEx   | 0.554 | 0.648 | 0.693 | 0.692 | 0.594   |
| DistMult  | 0.575 | 0.613 | 0.642 | 0.604 | 0.603   |
| HolE      | 0.557 | 0.488 | 0.639 | 0.602 | 0.733   |


| WN11      | Uncal | Platt | Isot  | beta  | histbin |
|-----------|-------|-------|-------|-------|---------|
| TransE    | 0.507 | 0.883 | 0.881 | 0.879 | 0.881   |
| ComplEx   | 0.559 | 0.590 | 0.623 | 0.623 | 0.601   |
| DistMult  | 0.566 | 0.602 | 0.632 | 0.631 | 0.617   |
| HolE      | 0.621 | 0.692 | 0.698 | 0.695 | 0.697   |


| YAGO39    | Uncal | Platt | Isot  | beta  | histbin |
|-----------|-------|-------|-------|-------|---------|
| TransE    | 0.506 | 0.695 | 0.720 | 0.698 | 0.690   |
| ComplEx   | 0.896 | 0.900 | 0.905 | 0.896 | 0.901   |
| DistMult  | 0.892 | 0.889 | 0.893 | 0.890 | 0.893   |
| HolE      | 0.834 | 0.852 | 0.852 | 0.850 | 0.852   |


| DBpedia50 | Uncal | Platt | Isot  | beta  | histbin |
|-----------|-------|-------|-------|-------|---------|
| TransE    | 0.500 | 0.858 | 0.876 | 0.870 | 0.875   |
| ComplEx   | 0.615 | 0.668 | 0.667 | 0.667 | 0.629   |
| DistMult  | 0.649 | 0.696 | 0.707 | 0.698 | 0.704   |
| HolE      | 0.645 | 0.743 | 0.753 | 0.731 | 0.753   |


| Nations   | Uncal | Platt | Isot  | beta  | histbin |
|-----------|-------|-------|-------|-------|---------|
| TransE    | 0.50  | 0.525 | 0.498 | 0.520 | 0.512   |
| ComplEx   | 0.43  | 0.418 | 0.567 | 0.463 | 0.587   |
| DistMult  | 0.49  | 0.522 | 0.520 | 0.532 | 0.570   |
| HolE      | 0.52  | 0.535 | 0.537 | 0.500 | 0.520   |


| Kinship   | Uncal | Platt | Isot  | beta  | histbin |
|-----------|-------|-------|-------|-------|---------|
| TransE    | 0.500 | 0.530 | 0.543 | 0.534 | 0.521   |
| ComplEx   | 0.712 | 0.741 | 0.804 | 0.792 | 0.796   |
| DistMult  | 0.868 | 0.891 | 0.894 | 0.892 | 0.895   |
| HolE      | 0.795 | 0.793 | 0.791 | 0.799 | 0.788   |


| UMLS      | Uncal | Platt | Isot  | beta  | histbin |
|-----------|-------|-------|-------|-------|---------|
| TransE    | 0.500 | 0.771 | 0.770 | 0.769 | 0.760   |
| ComplEx   | 0.838 | 0.893 | 0.899 | 0.898 | 0.902   |
| DistMult  | 0.850 | 0.880 | 0.884 | 0.884 | 0.884   |
| HolE      | 0.875 | 0.868 | 0.869 | 0.878 | 0.876   |



We use these probabilities to perform triple calissfication task. Diagram 3 shows that the probabilities can serve as a good indicator to classify the positve triples from the negative ones. It achieve STOA results as the literature standard of per-relation threshold. 

The devil is in the details. In all the above experiments, we obtain the calibrated probabilities by directly applying the calibration model on the embedding scores. What if we change the stratefy, to obtain calibrated probabilities by applying calibration model on the uncalibrated probabilities that were generated via expit transform?

we observe that the so called uncalibrated probabilities are hard to be recognised as probabilities.


## Discussion
