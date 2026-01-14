import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


def get_roc_scores(model, pos_g_l, neg_g_l, embed_l, ntype=None):
    model.eval()
    auc_scores, ap_scores = [], []
    for t in range(len(pos_g_l)):
        pos_g = pos_g_l[t]
        neg_g = neg_g_l[t]
        embed = embed_l[t]
        if ntype == None:
            pos_score = model(pos_g, embed).sigmoid().detach().cpu().numpy()
            neg_score = model(neg_g, embed).sigmoid().detach().cpu().numpy()
        else:
            pos_score = model(pos_g, embed[ntype]).sigmoid().detach().cpu().numpy()
            neg_score = model(neg_g, embed[ntype]).sigmoid().detach().cpu().numpy()
        preds_all = np.hstack([pos_score, neg_score])
        labels_all = np.hstack([np.ones(len(pos_score)), np.zeros(len(neg_score))])
        auc_scores.append(roc_auc_score(labels_all, preds_all))
        ap_scores.append(average_precision_score(labels_all, preds_all))
    model.train()
    return auc_scores, ap_scores


def get_mae_scores(model, glabel_l, out_feat_l):
    model.eval()
    with torch.no_grad():
        mae_list, rmse_list = [], []
        for glabel, feat in zip(glabel_l, out_feat_l):
            pred = model(feat['state'])
            label = glabel.nodes['state'].data['feat']
            loss = F.l1_loss(pred, label)
            rmse = torch.sqrt(F.mse_loss(pred, label))

            mae_list.append(loss.item())
            rmse_list.append(rmse.item())
        mae = sum(mae_list) / len(mae_list)
        rmse = sum(rmse_list) / len(rmse_list)
    model.train()
    return mae, rmse


def construct_negative_graph(g, n_neg):
    neg = {}
    for can_etype in g.canonical_etypes:
        if 'hyper' in can_etype[1]: continue
        n_edges = g.num_edges(etype=can_etype)
        src_neg, dst_neg = dgl.sampling.global_uniform_negative_sampling(
            g, n_edges*n_neg, etype=can_etype)
        neg[can_etype] = (src_neg, dst_neg)
    return dgl.heterograph(neg, num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes})


def construct_positive_graph(g, eratio=0.3):
    neg = {}
    for can_etype in g.canonical_etypes:
        if 'hyper' in can_etype[1]: continue
        n_edges = g.num_edges(etype=can_etype)
        src, dst = g.edges(etype=can_etype)
        ntrain = int(eratio*n_edges)
        perm = np.random.permutation(n_edges)
        neg[can_etype] = src[perm[:ntrain]], dst[perm[:ntrain]]
    return dgl.heterograph(neg, num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes})


def bce_loss(model, glist, feat_l, ratio, n_neg):
    loss = 0
    for glabel, feat in zip(glist, feat_l):
        pos_glabel = construct_positive_graph(glabel, ratio).to(glabel.device)
        neg_glabel = construct_negative_graph(glabel, n_neg).to(glabel.device)
        pos_score = model(pos_glabel, feat)
        neg_score = model(neg_glabel, feat)
        pred = torch.cat((pos_score, neg_score))
        label = torch.cat((torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])))
        loss += F.binary_cross_entropy_with_logits(pred, label.to(glabel.device))
    return loss