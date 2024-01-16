# Train MLP for GNNs

- paper
  link: [Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs](https://openreview.net/forum?id=dqnNW2omZL6),
  and [MLPInit: Embarrassingly Simple GNN Training Acceleration with MLP Initialization](https://openreview.net/forum?id=P8YIphWNEGO)
- official code repo: https://github.com/snap-research/MLPInit-for-GNNs, and https://github.com/chr26195/PMLP

## Dependencies

- torch
- dgl

## Implementation

- Version 1: Implementation based on Dropping all edges during training GNN.
- Version 2: Implementation based on defining new GNN layer, which allows to use weight matrix only.
- Run code: `python train_v1.py` `python train_v2.py`
- Note: all codes based on [DGL](https://github.com/dmlc/dgl)

## Summary

|      | cora  | pubmed | citeseer |
|:----:|:-----:|:------:|:--------:|
| MLP  | 0.565 |  0.71  |  0.552   |
| GCN  | 0.806 |  0.79  |  0.703   |
| PMLP | 0.763 | 0.752  |  0.631   |
