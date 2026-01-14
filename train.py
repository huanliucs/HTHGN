import statistics as st
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import bce_loss, get_roc_scores
from utils.pytorchtools import EarlyStopping
import utils.data as dt
from models.predictor import LinkPredictor
from models.hthgn import HTHGN
import datetime 

def run_hthgn(args, device):
    scores = []
    for _ in range(args.n_repeat):
        data = dt.load_data(args, device)
        glabel_l, feat_l, train_l, val_l, val_nl, test_l, test_nl, pos_l, neg_l, pos_l_new, neg_l_new = data
        
        # k-ring Uniform Heterogeneous Hyper Construction
        # H = (V, E)
        print(f'{args.k}-{args.hetype} {args.p}-uniform HTHG Constructing...')
        glabel_l = dt.HG_gen(args, device, glabel_l)
        print('Done!')
        
        in_dim = feat_l[0][glabel_l[0].ntypes[0]].shape[1]
        
        encoder = HTHGN(glabel_l[0], in_dim, args.n_hid, args.n_layers, args.n_heads, args.tw, norm=args.norm, device=device)
        decoder = LinkPredictor(n_inp=args.n_hid, n_classes=1)
        model = nn.Sequential(encoder, decoder)
        if args.use:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        else:
            model = model.to(device)
        
        data = glabel_l, feat_l, pos_l, neg_l, pos_l_new, neg_l_new
        ret = link_prediction(args, model, data)
        scores.append(ret)

    scores = np.array(scores)
    return scores.mean(0), scores.std(0), scores.min(0), scores.max(0)


def link_prediction(args, model, data):
    glabel_l, feat_l, pos_l, neg_l, pos_l_new, neg_l_new = data
    seq_len = len(glabel_l)
    seq_end = len(glabel_l) - args.len
    output = f'./output/{args.model}_{args.dataset}_{args.k}_{args.hntype}_{args.hetype}_{args.p}_{args.keepori}_{args.n_layers}_{args.n_hid}.pt'

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = ReduceLROnPlateau(optim, 'min', 0.5, patience=10, min_lr=1e-5)
    early_stopping = EarlyStopping(patience=args.patience, verbose=False, path=output)
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))
    logname = f'./output/logs/hyper/{args.model}_{args.dataset}_{args.k}_{args.hntype}_{args.hetype}/'+timestamp
    writer = SummaryWriter(log_dir=logname, comment=f'{args.model}_{args.dataset}_{args.k}_{args.hntype}_{args.hetype}')

    for epoch in range(args.epochs):
        model.train()
        out_feat_l = []
        loss_epoch = 0
        for left in range(seq_len-args.tw):
            out_feat = model[0](glabel_l[left:left+args.tw], feat_l=feat_l[left:left+args.tw])
            out_feat = {ntype: sum([out_feat[ntype][str(t)] for t in range(args.tw)]) for ntype in glabel_l[0].ntypes}
            out_feat_l.append(out_feat)
            loss = bce_loss(model[1], [glabel_l[args.tw+left]], [out_feat], args.eratio, args.n_neg)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_epoch += loss.item()
        
        writer.add_scalar("Pretrain Loss", loss_epoch, epoch)
        # scheduler.step(loss)
        print('Epoch: {:03} | Loss: {:.4f} | Lr: {:.6f} | Count: {:02}'.format(
            epoch, loss_epoch, optim.param_groups[0]['lr'], early_stopping.counter))

        early_stopping(loss, model)
        if early_stopping.early_stop or epoch == args.epochs-1:
            writer.close()
            model.load_state_dict(torch.load(output))
            model.eval()
            with torch.no_grad():
                # Prediction
                auc_l, ap_l = get_roc_scores(model[1], pos_l[seq_end:], neg_l[seq_end:], out_feat_l[seq_end-args.tw:])
                # Prediction New
                auc_new_l, ap_new_l = get_roc_scores(model[1], pos_l_new[seq_end:], neg_l_new[seq_end:], out_feat_l[seq_end-args.tw:])

                auc = st.mean(auc_l)
                ap = st.mean(ap_l)
                auc_new = st.mean(auc_new_l)
                ap_new = st.mean(ap_new_l)

                print('Prediction AUC/AP: {:.4f}/{:.4f} | Prediction New AUC/AP: {:.4f}/{:.4f}'.format(
                    auc, ap, auc_new, ap_new))
            break
    return auc, ap, auc_new, ap_new

