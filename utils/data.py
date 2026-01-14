import dgl
import numpy as np
import torch
import pickle
import os
import dgl

def load_detection(ori_l, eratio):
    glabel_l, train_l, val_l, val_nl, test_l, test_nl = [], [], [], [], [], []
    for g in ori_l:
        num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes}
        glabel, train, val, val_neg, test, test_neg = {}, {}, {}, {}, {}, {}
        for can_etype in g.canonical_etypes:
            n_edges = g.num_edges(etype=can_etype)
            ntrain = int(eratio*n_edges)
            nval = int((1+eratio)*n_edges/2)
            perm = np.random.permutation(n_edges)
            src, dst = g.edges(etype=can_etype)
            src_neg, dst_neg = dgl.sampling.global_uniform_negative_sampling(g, n_edges, etype=can_etype)
            # Bi-direction
            train[can_etype] = (src[perm[:ntrain]], dst[perm[:ntrain]])
            stype, etype, dtype = can_etype
            train[(dtype, etype+'_r', stype)] = (dst[perm[:ntrain]], src[perm[:ntrain]])
            # Single-direction
            val[can_etype] = (src[perm[ntrain:nval]], dst[perm[ntrain:nval]])
            val_neg[can_etype] = (src_neg[ntrain:nval], dst_neg[ntrain:nval])
            test[can_etype] = (src[perm[nval:]], dst[perm[nval:]])
            test_neg[can_etype] = (src_neg[nval:], dst_neg[nval:])
            # Origin Bi-direction
            glabel[can_etype] = (src, dst)
            glabel[(dtype, etype+'_r', stype)] = (dst, src)

        glabel_l.append(dgl.heterograph(glabel, num_nodes_dict=num_nodes_dict, device=g.device))
        train_l.append(dgl.heterograph(train, num_nodes_dict=num_nodes_dict, device=g.device))
        val_l.append(dgl.heterograph(val, num_nodes_dict=num_nodes_dict, device=g.device))
        val_nl.append(dgl.heterograph(val_neg, num_nodes_dict=num_nodes_dict, device=g.device))
        test_l.append(dgl.heterograph(test, num_nodes_dict=num_nodes_dict, device=g.device))
        test_nl.append(dgl.heterograph(test_neg, num_nodes_dict=num_nodes_dict, device=g.device))

    return glabel_l, train_l, val_l, val_nl, test_l, test_nl


def load_prediction(ori_l, test_ratio=0.3):
    pos_l, neg_l = [], []
    for t in range(len(ori_l)):
        g = ori_l[t]
        num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes}
        neg = {}
        for can_etype in g.canonical_etypes:
            n_edges = g.num_edges(etype=can_etype)
            src_neg, dst_neg = dgl.sampling.global_uniform_negative_sampling(g, n_edges, etype=can_etype)
            neg[can_etype] = (src_neg, dst_neg)
            ntrain = int(test_ratio*n_edges)
            perm = np.random.permutation(n_edges)
            neg[can_etype] = src_neg[perm[:ntrain]], dst_neg[perm[:ntrain]]
        pos_l.append(g)
        neg_l.append(dgl.heterograph(neg, num_nodes_dict=num_nodes_dict, device=g.device))

    return pos_l, neg_l


def load_prediction_new(ori_l, test_ratio=0.3):
    pos_l, neg_l = [], []

    g = ori_l[0]
    num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    neg = {}
    for can_etype in g.canonical_etypes:
        n_edges = g.num_edges(etype=can_etype)
        src_neg, dst_neg = dgl.sampling.global_uniform_negative_sampling(g, n_edges, etype=can_etype)
        neg[can_etype] = (src_neg, dst_neg)
    pos_l.append(g)
    neg_l.append(dgl.heterograph(neg, num_nodes_dict=num_nodes_dict, device=g.device))

    for t in range(1, len(ori_l)):
        g = ori_l[t]
        num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes}
        pos, neg = {}, {}
        for can_etype in g.canonical_etypes:
            cur_g = g[can_etype].adj().to_dense()
            pri_g = ori_l[t-1][can_etype].adj().to_dense()
            new_g = np.transpose(np.asarray(np.where((cur_g - pri_g).cpu() > 0)))
            n_edges = new_g.shape[0]
            src_neg, dst_neg = dgl.sampling.global_uniform_negative_sampling(g, n_edges, etype=can_etype)
            pos[can_etype] = (new_g[:, 0], new_g[:, 1])
            neg[can_etype] = (src_neg, dst_neg)
        pos_l.append(dgl.heterograph(pos, num_nodes_dict=num_nodes_dict, device=g.device))
        neg_l.append(dgl.heterograph(neg, num_nodes_dict=num_nodes_dict, device=g.device))

    return pos_l, neg_l


def load_link(args, device):
    ori_l, _ = dgl.load_graphs(f'./data/{args.dataset}.bin')
    ori_l = [g.to(device) for g in ori_l]
    glabel_l, train_l, val_l, val_nl, test_l, test_nl = load_detection(ori_l, args.eratio)
    pos_l, neg_l = load_prediction(ori_l)
    pos_l_new, neg_l_new = load_prediction_new(ori_l)

    feat_l, _ = dgl.load_graphs(f'./data/mp2vec/{args.dataset}_mpfeat.bin')
    feat_l = [g.to(device) for g in feat_l]
    feat_l = [g.ndata['mp_feat'] for g in feat_l]
    
    return glabel_l, feat_l, train_l, val_l, val_nl, test_l, test_nl, pos_l, neg_l, pos_l_new, neg_l_new


def load_data(args, device):
    if args.dataset in ['yelp', 'aminer', 'dblp']:
        data = load_link(args, device)

    return data


def HG_gen(args, device, g_list):
    
    filename = f'./data/dataset/Hyper_{args.dataset}_{args.k}_{args.hntype}_{args.hetype}_{args.p}_{args.keepori}.pkl'

    if os.path.exists(filename):
        # 如果文件存在，则读取它
        with open(filename, 'rb') as file:
            pg_list = pickle.load(file)
    else:
        # 如果文件不存在，则保存HG为HG.pkl
        g_list = [g.to('cpu') for g in g_list]
        pg_list = []
        for i, g in enumerate(g_list):
            print(f'Constructing {i}-th HyperGraph on {g.device}...')
            pg_list.append(HG_generator(g, k=args.k, p=args.p, keep_ori=args.keepori, hntype=args.hntype, hetype=args.hetype, 
                                    copy_ndata=False, copy_edata=False))

        with open(filename, 'wb') as file:
            pickle.dump(pg_list, file)
    pg_list = [g.to(device) for g in pg_list]
    
    return pg_list


def HG_generator(g, k=3, p=100, keep_ori=True, hntype='_all', hetype='hop', copy_ndata=True, copy_edata=True):
    num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    if hntype == '_all':
        for ntype in g.ntypes:
            num_nodes_dict['hyper_' + ntype] = g.num_nodes(ntype)
    else:
        num_nodes_dict['hyper_' + hntype] = g.num_nodes(hntype)
    
    homog = dgl.to_homogeneous(g)
    if hetype == 'hop':
        khopg = dgl.to_simple(dgl.merge([dgl.khop_graph(homog, i) for i in range(1, k+1)]))
    elif hetype == 'ring':
        khopg = dgl.to_simple(dgl.khop_graph(homog, k))

    khopg.edata['_ID'] = torch.arange(0, khopg.num_edges())
    khopg.edata['_TYPE'] = torch.ones(khopg.num_edges(), dtype=homog.idtype) * len(g.etypes)

    hyper_ntypes, hyper_etypes = g.ntypes.copy(), g.etypes.copy()
    hyper_etypes.append('hyperedge')
    khop_hg = dgl.to_heterogeneous(khopg, ntypes=hyper_ntypes, etypes=hyper_etypes)
    if p > 0:
        khop_hg = dgl.sampling.sample_neighbors(khop_hg, {ntype: khop_hg.nodes(ntype) for ntype in khop_hg.ntypes}, p)
    
    # Origin graph structure
    if keep_ori:
        graph_data = {cetype: g.edges(etype=cetype) for cetype in g.canonical_etypes}
    else:
        graph_data = {}

    for stype, etype, dtype in khop_hg.canonical_etypes:
        if hntype == '_all' or stype == hntype:
            src, dst = khop_hg.edges(etype = (stype, etype, dtype))
            graph_data['hyper_'+stype, etype, dtype] = (src, dst)
            graph_data[dtype, etype+'_r', 'hyper_'+stype] = (dst, src)
        elif dtype == hntype:
            src, dst = khop_hg.edges(etype = (stype, etype, dtype))
            graph_data[stype, etype, 'hyper_'+dtype] = (src, dst)
            graph_data['hyper_'+dtype, etype+'_r', stype] = (dst, src)

    khop_hg = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    
    # handle ndata
    if copy_ndata:
        # for each ntype
        for ntype in g.ntypes:
            khop_hg.nodes[ntype].data.update(g.nodes[ntype].data)

    # handle edata
    if copy_edata:
        # for each etype
        for utype, etype, vtype in g.canonical_etypes:
            khop_hg.edges[utype, etype, vtype].data.update(
                g.edges[utype, etype, vtype].data
            )
    
    return khop_hg
