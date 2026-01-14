import argparse
import sys

def set_params():
    argv = sys.argv
    model_name = argv[1]

    parser = argparse.ArgumentParser(description='HTG')

    parser.add_argument("--dataset", type=str, default='yelp', 
        choices=('dblp', 'aminer', 'yelp'), help="dataset")
    parser.add_argument("--file", type=str, default='log.txt', help="output file")
    parser.add_argument("--n_hid", type=int, default=32, help="hidden dim")
    parser.add_argument("--eratio", type=float, default=1.0, help="training edges ratio")
    parser.add_argument("--nratio", type=float, default=0.8, help="training nodes ratio")
    parser.add_argument("--gpu", type=int, default=-1, help="indx of GPU")
    parser.add_argument('--use', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1',help='device ids of multile gpus')
    parser.add_argument("--seed", type=int, default=0, help="rand seed")
    parser.add_argument("--n_repeat", type=int, default=1, help="repeat times")
    parser.add_argument("--len", type=int, default=3, help="predict len")
    parser.add_argument("--n_neg", type=int, default=5, help="negative edges")
    parser.add_argument("--patience", type=int, default=30, help="early stopping")
    parser.add_argument("--tw", type=int, default=3, help="time_window")

    parser.add_argument("--model", type=str, default='hthgn', help="model")
    parser.add_argument("--n_inp", type=int, default=32, help="GNN inp dim")
    parser.add_argument("--n_layers", type=int, default=2, help="THEG layers")
    parser.add_argument("--n_heads", type=int, default=4, help="attention head")
    parser.add_argument("--n_classes", type=int, default=1, help="classes")
    parser.add_argument("--k", type=int, default=3, help="range number")
    parser.add_argument("--p", type=int, default=100, help="range number")
    parser.add_argument("--keepori", type=bool, default=True, help="keep low order edges")
    parser.add_argument("--hetype", type=str, default='hop', help="hyper edge type")
    parser.add_argument("--hntype", type=str, default='_all', help="hyper node type")
    parser.add_argument("--norm", type=bool, default=False, help="norm on output layer")
    parser.add_argument("--epochs", type=int, default=2000, help="train epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="lr")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")

    args, _ = parser.parse_known_args()

    return args

