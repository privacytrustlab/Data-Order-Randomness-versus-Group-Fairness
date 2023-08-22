import torch
import os

from train_utils import train_mlp
from models import get_model, get_experiment_fldr
from dataloader import get_dataset
from argparse_helper import get_args

torch.autograd.set_detect_anomaly(True)

def train(args):
    torch.manual_seed(args.init_seed)

    model = get_model(args.arch, args.num_feat, ckpt=None, cuda=args.cuda, dropout=args.dropout)
    savefldr = get_experiment_fldr(args.ckptfldr, args.pc, args.loss, args.arch, args.init_seed, args.shuffle_seed)

    trainloader, validloader, testloader = get_dataset(dataset=args.dataset, protected_class=args.pc,
                                                       shuffle_seed=args.shuffle_seed, batch_size=args.batch_size,
                                                       fairbatch=args.fairbatch, model=model)

    train_mlp(model, trainloader, validloader, savefldr, args)

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    train(args)
