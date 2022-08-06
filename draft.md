# REvisiting Probability Calibration in Knowledge Graph Embedding

## Introduction



1. We reprecated and extended the previous experiments. We confirmed their conclusion but also pointed out the potential flaws of previous works:
    - Calibration techqniues like Platt Scaling, Isotonic Regression can effectively convert embedding scores into good probabilities.
    - However, converting embedding scores via expit transform (simply pss the scores through a sigmoid layer) is not a good idea, because the transformed probabilities are not only uncalibrated, but also can't be used by calibration. Arguably, some may even refuse to recognise these transformed values as probabilities. A better idea is to convert the embedding scores to probabilities directly via platt scaling or isontonic regression, instead of expit transform.
2. We also found a more useful rule of thumb about how to choose calibration techqniue: when we have a large set of held-out data (over 10 thousand triples), we should definitely choose binning-based calibration techniques, such as Isonotic Regression, otherwise scaling-based techqniues, such as Platt Scaling, would be more suitable. 


## From Embedding Scores to Probabilities


## Results

In our experiment, we train 4 typical KE embedding models, TransE, ComplEx, DistMult, and HoLE on 7 datasets: FB13k, WN11, YAGO39, DBpedia50, Nations, Kinship, UMLS. Each dataset is splited into 3 subset for train, valid, and test. Note that the valid and test set of FB13, WN11 and YAGO39 have ground truth negative samples, while the other 4 don't have. Thus, we generate synthetic negative samples via the corrption and local closed world assumtion. Statistics of the datasets are summarisd in table 1.

[table 1]

We run calibration techniques for every KG Embedding models. During experiment, we train the KGE models only use the train set, and train the calibration model using valid set. In diagram 1, the calibration curves illustrate that all calibration techqniues produced better-calibrated probabilities than those obtained via expit transform.

[diagram 1]

We can see that binning-based calibration performs better in general. We also noticed that binning-based methods dominate in FB13k, WN11 and YAGO39. We hypothesise that perhaps that's because there are more data in these 3 datasets. Previous work also suggested that binnig-based methods tend to overfit, expecially in smaller datasets. Thus, we take these 3 datasets, and gradaully shrink the size of valid set by randomly samping k%, and compare the number of winning between binning-based and scaling-based methods. 

[Diagram 2]

Diagram 2 shows that as the size of the dataset shrink, the winning count of binning-based methods decreases while caling-based method increases, i.e., scaling-based calibration start to gain better calibration peroformance than binning-based techqniue in some cases.

[Diagram 3]

We use these probabilities to perform triple calissfication task. Diagram 3 shows that the probabilities can serve as a good indicator to classify the positve triples from the negative ones. It achieve STOA results as the literature standard of per-relation threshold. 

The devil is in the details. In all the above experiments, we obtain the calibrated probabilities by directly applying the calibration model on the embedding scores. What if we change the stratefy, to obtain calibrated probabilities by applying calibration model on the uncalibrated probabilities that were generated via expit transform?

[Diagram 4]

Diagram 4 illustrate that the calibration techniques all work well.

[Diagram 5] 

However, these calibrated probabilities get poor results in triple classification tasks.



## Discussion
