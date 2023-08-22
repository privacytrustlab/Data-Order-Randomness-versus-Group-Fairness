import torch
import os
import numpy as np
from tqdm import tqdm

from train_utils import test_mlp
from models import get_model, get_experiment_fldr
from dataloader import get_dataset
from argparse_helper import get_args

torch.autograd.set_detect_anomaly(True)

def fairness_variance_boxplot(seed_tuple_list, args):

    trainloader, validloader, testloader = get_dataset(dataset=args.dataset, protected_class=args.pc,
                                                       shuffle_seed=args.shuffle_seed, batch_size=args.batch_size)

    fscore_arr, avg_odds_arr = [], []
    for (init_seed, shuffle_seed) in tqdm(seed_tuple_list):
        savefldr = get_experiment_fldr(args.ckptfldr, args.pc, args.loss, args.arch, init_seed, shuffle_seed)
        model = get_model(args.arch, args.num_feat, ckpt=savefldr + '299.pth', cuda=args.cuda, dropout=args.dropout)

        fscore, avg_odds = test_mlp(model, testloader, cuda=args.cuda, fairness_criteria='avgodds')
        fscore_arr.append(fscore)
        avg_odds_arr.append(avg_odds)

    print("Metric \t\t Min \t 25th Percentile \t Median \t 75th Percentile \t Max")
    print("F-Score \t %f \t %f \t %f \t %f \t %f" % (np.min(fscore_arr), np.percentile(fscore_arr, 25), np.median(fscore_arr), np.percentile(fscore_arr, 75), np.max(fscore_arr)))
    print("Avg Odds \t %f \t %f \t %f \t %f \t %f" % (np.min(avg_odds_arr), np.percentile(avg_odds_arr, 25), np.median(avg_odds_arr), np.percentile(avg_odds_arr, 75), np.max(avg_odds_arr)))

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    seed_tuple_list = []
    for s in range(50):
        init_seed, shuffle_seed = s, s
        seed_tuple_list.append((init_seed, shuffle_seed))

    fairness_variance_boxplot(seed_tuple_list, args)
