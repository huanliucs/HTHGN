import torch
import torch.nn as nn
import dgl
import math
import numpy as np
import torch.nn.functional as F
import dgl.nn as dglnn

class TempAttention(nn.Module):
    def __init__(self, n_inp: int, n_hid: int, time_window: int, device: torch.device):
        super(TempAttention, self).__init__()

        self.proj = nn.Linear(n_inp, n_hid)
        self.q_w  = nn.Linear(n_hid, n_hid, bias = False)
        self.k_w  = nn.Linear(n_hid, n_hid, bias = False)
        self.v_w  = nn.Linear(n_hid, n_hid, bias = False)
        self.fc   = nn.Linear(n_hid, n_hid)
        self.pe   = torch.tensor(self.generate_positional_encoding(n_hid, time_window)).float().to(device)

    def generate_positional_encoding(self, d_model, max_len):
        pe = np.zeros((max_len, d_model))
        for i in range(max_len):
            for k in range(0, d_model, 2):
                div_term = math.exp(k * - math.log(100000.0) / d_model)
                pe[i][k] = math.sin( (i + 1) * div_term)
                try:
                    pe[i][k + 1] = math.cos( (i + 1) * div_term)
                except:
                    continue
        return pe
   
    def forward(self, x):
        x = x.permute(1,0,2)
        h = self.proj(x)
        h = h + self.pe
        q = self.q_w(h)
        k = self.k_w(h)
        v = self.v_w(h)
       
        qk = torch.matmul(q, k.permute(0, 2, 1))
        score = F.softmax(qk, dim = -1)

        h_ = torch.matmul(score, v)
        h_ = F.relu(self.fc(h_))
        
        return h_


class HeteAttention(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 ntypes,
                 num_bases,
                 *,
                 num_heads=4,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(HeteAttention, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.batchnorm = False
        self.ntypes = ntypes

        self.n_hid = out_feat // num_heads
        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GATv2Conv(in_feat, self.n_hid, num_heads=num_heads)
            for rel in rel_names
        }, self._rel_agg)

        self.proj = nn.Sequential(
                    nn.Linear(out_feat, out_feat),
                    nn.Tanh(),
                    nn.Linear(out_feat, 1, bias=False)
                )
        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))
        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_feat)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=gain)

    def _rel_agg(self, tensors, dsttype):
        h = torch.stack(tensors, dim=1)
        h = h.flatten(-2)
        w = self.proj(h).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((h.shape[0],) + beta.shape)
        h_ = (beta * h).sum(1)
        
        return F.relu(h_)

    
    def forward(self, g, inputs):
        g = g.local_var()
        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs_src)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class HTHGNLayer(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, n_heads: int, 
                 timeframe, norm: bool, device: torch.device, dropout: float):
        super(HTHGNLayer, self).__init__()

        self.n_inp     = n_inp
        self.n_hid     = n_hid
        self.n_heads   = n_heads
        self.timeframe = timeframe
        self.norm      = norm
        self.dropout   = dropout
        self.device    = device
        self.ntypes = graph.ntypes
        self.hg_list = None
        self.node_dict = {}
        self.edge_dict = {}
        for ntype in graph.ntypes:
            self.node_dict[ntype] = len(self.node_dict)
        for etype in graph.canonical_etypes:
            self.edge_dict[etype] = len(self.edge_dict)

        self.rel_names = list(set(graph.etypes))
        self.rel_names.sort()
        self.num_bases = len(self.rel_names)

        # Heterogeneous Attention
        self.rgcn_agg = HeteAttention(
            n_hid, n_hid, self.rel_names, self.ntypes,
            self.num_bases, activation=F.relu, self_loop=False,
            dropout=self.dropout, weight=True)

        # Temporal Attention
        self.cross_time_agg = nn.ModuleDict({
            ntype: TempAttention(n_hid, n_hid, len(timeframe), device)
            for ntype in graph.ntypes
        })

        # Gate mechanism
        self.res_fc = nn.ModuleDict()
        self.res_weight = nn.ParameterDict()
        for ntype in graph.ntypes:
            self.res_fc[ntype] = nn.Linear(n_inp, n_hid)
            self.res_weight[ntype] = nn.Parameter(torch.randn(1))

        self.reset_parameters()
        
        # LayerNorm
        if norm:
            self.norm_layer = nn.ModuleDict({ntype: nn.LayerNorm(n_hid) for ntype in graph.ntypes})

    def reset_parameters(self):
        """Reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        for ntype in self.res_fc:
            nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)

    def forward(self, g_l, feat_l):
        inter_features = dict({ntype:{} for ntype in self.ntypes})
        for t, graph in enumerate(g_l):
            ttype = str(t)
            feat = {ntype: feat_l[ntype][ttype] for ntype in graph.ntypes}
            out_feat = self.rgcn_agg(graph, feat)
            for ntype in graph.ntypes:
                inter_features[ntype][ttype] = out_feat[ntype]
        
        output_features = {}
        for ntype in inter_features:
            output_features[ntype] = {}
            out_emb = [inter_features[ntype][ttype] for ttype in inter_features[ntype]]
            time_embeddings = torch.stack(out_emb, dim=0)
            h = self.cross_time_agg[ntype](time_embeddings).permute(1,0,2)
            output_features[ntype] = {ttype: h[i] for (i, ttype) in enumerate(self.timeframe)}

        new_features = {}
        for ntype in output_features:
            new_features[ntype] = {}
            alpha = torch.sigmoid(self.res_weight[ntype])
            for ttype in self.timeframe:
                new_features[ntype][ttype] = output_features[ntype][ttype] * alpha + self.res_fc[ntype](feat_l[ntype][ttype]) * (1 - alpha)
                if self.norm:
                    new_features[ntype][ttype] = self.norm_layer[ntype](new_features[ntype][ttype])

        return new_features


class HTHGN(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, n_layers: int, 
                 n_heads: int, time_window: int, norm: bool, device: torch.device, dropout: float = 0.2):
        super(HTHGN, self).__init__()

        self.n_inp     = n_inp
        self.n_hid     = n_hid
        self.n_layers  = n_layers
        self.n_heads   = n_heads
        self.device = device
        self.timeframe = [str(i) for i in range(time_window)]

        self.adaption_layer = nn.ModuleDict({ntype: nn.Linear(n_inp, n_hid) for ntype in graph.ntypes})
        self.gnn_layers     = nn.ModuleList([HTHGNLayer(graph, n_hid, n_hid, n_heads, 
                                                        self.timeframe, norm, device, dropout) for _ in range(n_layers)])
    
    def forward(self, g_l, feat_l):
        assert len(g_l) == len(feat_l)
        assert len(g_l) == len(self.timeframe)
        
        inp_feat = {ntype: {} for ntype in g_l[0].ntypes}
        for t, feat in enumerate(feat_l):
            ttype = str(t)
            for ntype in g_l[t].ntypes:
                if ntype in feat:
                    inp_feat[ntype][ttype] = self.adaption_layer[ntype](feat[ntype])
                else:
                    inp_feat[ntype][ttype] = torch.zeros((g_l[t].num_nodes(ntype=ntype), self.n_hid), device=self.device)

        for i in range(self.n_layers):
            inp_feat = self.gnn_layers[i](g_l, inp_feat)
    
        return inp_feat


