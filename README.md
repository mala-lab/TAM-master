>📋  A template README.md for code accompanying a Machine Learning paper

# Our Paper Title

This repository is the official implementation of Truncated Affinity Maximization: One-class
Homophily Modeling for Graph Anomaly Detection 

The full paper can be found at [NeurIPS Portal](https://nips.cc/virtual/2023/poster/70486) or [arXiv](https://arxiv.org/pdf/2306.00006.pdf).



[//]: # ( Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials)
>📋  We explore the  property one class homophily to introduce a novel unsupervised anomaly scoring measure for GAD -- local node affinity -- that assigns a larger anomaly score to nodes that are less affiliated with their neighbors, with the affinity defined as similarity on node attributes/representations.
We further propose Truncated Affinity Maximization (TAM) that learns tailored node representations for our anomaly measure by maximizing the local affinity of nodes to their neighbors.
TAM is instead optimized on truncated graphs where non-homophily edges are removed iteratively to mitigate this bias. Extensive empirical results on six real-world GAD datasets show that TAM substantially outperforms seven competing models
>
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

[//]: # (Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...)
>📋  TAM is implemented in Pytorch 1.6.0 with python 3.7 and all the experiments are run on an NVIDIA GeForce RTX 3090 24GB GPU.

## Datasets
> BlogCatalog and ACM were downloaded from https://github.com/yixinliu233/CoLA 
> Amazon and Yelpchi were downloaded from  https://github.com/YingtongDou/CARE-GNN
> Amazon-all and Yelpchi-all were downloaded from  https://github.com/YingtongDou/CARE-GNN
> Facebook is obtained from  https://github.com/zhiming-xu/conad
> Reddit is downloaded from https://github.com/pygod-team/data
> T-finance is downloaded from https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr
> OGB-Protein is downloaded from https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv


Note that considering it's difficult to conduct an evaluation on the isolated nodes in the graph, they were removed before modeling.
## Training

To train the model(s) in the paper, run this command:

```train
python train.py
```

>📋  In TAM, each LAMNet is implemented by a two-layer GCN, and its weight parameters are optimized using Adam optimizer with 500 epochs and a learning rate of 1e-5 by default. 
T=3 and K=4 are used for all datasets. Datasets with injected anomalies, such as BlogCatalog and ACM, require strong regularization, so $\lambda=1$ is used by default; whereas $\lambda=0$ is used for the four real-world datasets.
For the larger datasets like Amazon-all, YelpChi-all and OGB-Protein, they require larger truncation times  due to the large number of edges. So we set K = 8.
## Evaluation

To evaluate our model on dataset,  run:

```eval
python train.py 
```
>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below). \\
Since the setting is unsupervised, the model can be evaluated on the whole dataset, the training and evaluation run concurrently and iteratively.
## Results
Our model achieves the following performance on :

| Metric | BlogCatalog | ACM      | Amazon         | Facebook | Reddit | YelpChi |
|--------|-------------|----------|----------------|----------|--------|--------|
| AUROC  | 0.8248      | 0.8878   | 0.7064         | 0.9144   | 0.6023 | 0.5643 |
| AUPRC  | 0.4182      | 0.5124   | 0.2634   | 0.2233  | 0.0446 | 0.0778 |
>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing
We reveals an important anomaly-discriminative property, the one-class homophily, in GAD datasets with either injected or real anomalies. We utilize this property to introduce a novel unsupervised GAD measure, local node affinity, and further introduce a truncated affinity maximization
(TAM) approach that end-to-end optimizes the proposed anomaly measure on truncated adjacency  matrix with the non-homophily edges eliminated



>📋  Pick a licence and describe how to contribute to your code repository. 

If you use this package and find it useful, please cite our paper using the following BibTeX. Thanks! :)

```
@article{hezhe2023truncated,
  title={Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection},
  author={Hezhe, Qiao and Guansong, Pang},
  journal={arXiv preprint arXiv:2306.00006},
  year={2023}
}
```

## Acknowledgement and Reference
[1] Liu, Yixin, et al. "Anomaly detection on attributed networks via contrastive self-supervised learning." IEEE transactions on neural networks and learning systems 33.6 (2021): 2378-2392.
[2] Dou, Yingtong, et al. "Enhancing graph neural network-based fraud detectors against camouflaged fraudsters." Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2020.
[3] Liu, Kay, et al. "Pygod: A python library for graph outlier detection." arXiv preprint arXiv:2204.12095 (2022).
[4] Xu, Zhiming, et al. "Contrastive attributed network anomaly detection with data augmentation." Advances in Knowledge Discovery and Data Mining: 26th Pacific-Asia Conference, PAKDD 2022, Chengdu, China, May 16–19, 2022, Proceedings, Part II. Cham: Springer International Publishing, 2022.
[5] Hu, Weihua, et al. "Open graph benchmark: Datasets for machine learning on graphs." Advances in neural information processing systems 33 (2020): 22118-22133.
[6] Tang, Jianheng, et al. "Rethinking graph neural networks for anomaly detection." International Conference on Machine Learning. PMLR, 2022.
[7] Tang, Jianheng, et al. "GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection." arXiv preprint arXiv:2306.12251 (2023).