import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn

class LinkPredictor(nn.Module):
    def __init__(self, n_inp: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(n_inp * 2, n_inp)
        self.fc2 = nn.Linear(n_inp, n_classes)

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)
        y = self.fc2(F.relu(self.fc1(x)))
        return {'score': y}

    def forward(self, graph: dgl.DGLGraph, node_feat: dict, ntype=None):
        with graph.local_scope():
            if ntype == None:
                # graph.ndata['h'] = node_feat
                nfeat = {}
                for gntype in graph.ntypes:
                    nfeat[gntype] = node_feat[gntype]
                graph.ndata['h'] = nfeat
            else:
                graph.ndata['h'] = node_feat[ntype]
            score = []
            for can_etype in graph.canonical_etypes:
                graph.apply_edges(self.apply_edges, etype=can_etype)
                score.append(graph.edges[can_etype].data['score'])

            return torch.cat(score).flatten()


class DotProductPredictor(nn.Module):
    def forward(self, graph: dgl.DGLGraph, node_feat: dict):
        with graph.local_scope():
            graph.ndata['h'] = node_feat
            score = []
            for can_etype in graph.canonical_etypes:
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=can_etype)
                score.append(graph.edges[can_etype].data['score'])

            return torch.cat(score).flatten()


class HomoDotProductPredictor(nn.Module):
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class NodePredictor(nn.Module):
    def __init__(self, n_inp: int, n_classes: int):
        """
    
        :param n_inp      : int, input dimension
        :param n_classes  : int, number of classes
        """
        super().__init__()

        self.fc1 = nn.Linear(n_inp, n_inp)
        self.fc2 =  nn.Linear(n_inp, n_classes)

    def forward(self, node_feat: torch.tensor):
        """
        
        :param node_feat: torch.tensor
        """

        node_feat = F.relu(self.fc1(node_feat))
        pred = F.relu(self.fc2(node_feat))
        
        return pred