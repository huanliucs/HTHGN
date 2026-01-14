# Heterogeneous Temporal Hypergraph Neural Network

## Abstract: 
Graph representation learning (GRL) has emerged as an effective technique for modeling graph-structured data. When modeling the heterogeneity and dynamics in real-world complex networks, GRL methods designed for complex heterogeneous temporal graphs (HTGs) have been proposed and have achieved successful applications in various fields. However, most existing GRL methods mainly focus on preserving the low-order topology information while ignoring higher-order group interaction relationships, which are more consistent with real-world networks. Besides, most existing hypergraph methods can only model static homogeneous graphs, limiting their ability to model high-order interaction in HTGs. Therefore, to simultaneously enable the GRL model to capture high-order interaction relationships in HTGs, we first propose a formal definition of heterogeneous temporal hypergraphs and $P$-uniform heterogeneous hyperedges construction algorithm that do not rely on additional information. Then, a novel \underline{H}eterogeneous \underline{T}emporal \underline{H}yper\underline{G}raph \underline{N}eural network (HTHGN), is proposed to fully capture the higher-order interactions in HTGs. HTHGN contains a hierarchical attention mechanism module that simultaneously performs temporal message-passing between heterogeneous nodes and hyperedges to capture rich semantics in a wider receptive field brought by hyperedges. Furthermore, HTHGN performs contrastive learning by maximizing the consistency between low-order correlated heterogeneous node pairs on HTG to avoid the low-order structural ambiguity issue. Detailed experimental results on three real-world HTG datasets verify the effectiveness of the proposed HTHGN for modeling high-order interactions in HTGs and demonstrate significant performance improvements for HTG representation learning.

## Requirements
```
dgl==2.1.0+cu117
matplotlib==3.8.4
networkx==3.1
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
scipy==1.13.0
seaborn==0.13.2
tensorboard==2.12.1
torch==2.0.1
```

## Useage
```bash
python main.py hthgn --dataset yelp --gpu 0 --hetype "ring" --hntype "u" 
python main.py hthgn --dataset dblp --gpu 0 --hetype "ring" --hntype "a" 
python main.py hthgn --dataset aminer --gpu 0 --hetype "ring" --hntype "a"
```
