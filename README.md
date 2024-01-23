# Implementation of  Generalized Structural and Positional Encoding

<img src="https://latex.codecogs.com/gif.latex?\alpha-SPE" />
learn a joint node encoding that preserved both local node structural and global graph
structure in Graph Transformer

## Install
- The code is run with the `dgl` library (https://docs.dgl.ai/).
- The code base on the benchmarking graph neural network repo (https://github.com/graphdeeplearning/benchmarking-gnns.git).
- Due to the legacy code of `dgl`, several code base is run on `dgl==0.9.1` and others run on `dgl=1.0.0`
with or without GPU supports
- Support baseline:
    - Standard GNN models (GCN, GAT):
    - Graph Transformer based models: 
 
