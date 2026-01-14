import dgl
import warnings
import numpy as np
import torch
from utils.params import set_params
import train

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = set_params()

    print(args)
    dgl.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available() and args.use:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        device = torch.device(f'cuda:{args.gpu}')
    elif args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    train = train.run_hthgn

    ret = train(args, device)
    mean, std, mini, maxm = ret

    with open(args.file, 'a+') as f:
        print(args, file=f)
        print(mean[0], std[0], mean[1], std[1], file=f)
        print(mini[0], maxm[0], mini[1], maxm[1], file=f)
        print(mean[2], std[2], mean[3], std[3], file=f)
        print(mini[2], maxm[2], mini[3], maxm[3], file=f)
