import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

from train_utils import train_mlp, test_mlp
from models import get_model, get_experiment_fldr
from dataloader import get_dataset
from argparse_helper import get_args

torch.autograd.set_detect_anomaly(True)

def dataloader_to_numpy(dataloader):
    all_inputs, all_labels, all_groups = [], [], []
    for data in dataloader:
        inputs, labels, groups = data
        inputs, labels, groups = inputs.cpu().detach().numpy(), labels.cpu().detach().numpy(), groups.cpu().detach().numpy()

        all_inputs.extend(inputs)
        all_labels.extend(labels)
        all_groups.extend(groups)

    return np.array(all_inputs), np.array(all_labels), np.array(all_groups)

def subset_group(all_data, all_groups, all_labels, group, label):
    return all_data[all_groups==group][all_labels[all_groups==group]==label]

def make_trainloader_ratio(trainloader, ratio_dict, batch_size):

    all_inputs, all_labels, all_groups = dataloader_to_numpy(trainloader)
    minority_finished_flag = False

    ## Get All Subgroups Separately
    subset_dict = {}
    for subgroup in ratio_dict:
        group, label = subgroup[0], subgroup[1]

        inputs_sub = subset_group(all_inputs, all_groups, all_labels, group, label)
        labels_sub = subset_group(all_labels, all_groups, all_labels, group, label)
        groups_sub = subset_group(all_groups, all_groups, all_labels, group, label)

        subset_dict[subgroup] = (inputs_sub, labels_sub, groups_sub)

    ## Start Creating the New Data Order
    new_inputs, new_labels, new_groups = [], [], []
    for i in range(0, len(all_inputs), batch_size):
        batch_inputs, batch_labels, batch_groups = [], [], []
        for subgroup in ratio_dict:
            subgroup_ratio = ratio_dict[subgroup]
            inputs_sub, labels_sub, groups_sub = subset_dict[subgroup]

            start_index, end_index = int(subgroup_ratio*i), int(subgroup_ratio*(i+batch_size))
            if end_index > len(inputs_sub):
                minority_finished_flag = True
                break

            batch_inputs.extend(inputs_sub[start_index:end_index])
            batch_labels.extend(labels_sub[start_index:end_index])
            batch_groups.extend(groups_sub[start_index:end_index])

        ## We only create data order till the smallest group is finished. Beyond that, everything is again arranged randomly
        if minority_finished_flag:
            left_inputs, left_labels, left_groups = [], [], []
            for subgroup in ratio_dict:
                subgroup_ratio = ratio_dict[subgroup]
                inputs_sub, labels_sub, groups_sub = subset_dict[subgroup]

                start_index, end_index = int(subgroup_ratio*i), int(subgroup_ratio*(i+batch_size))

                left_inputs.extend(inputs_sub[start_index:])
                left_labels.extend(labels_sub[start_index:])
                left_groups.extend(groups_sub[start_index:])

            left_inputs, left_labels, left_groups = np.array(left_inputs), np.array(left_labels), np.array(left_groups)
            left_inputs, left_labels, left_groups = shuffle(left_inputs, left_labels, left_groups, random_state=0)

            new_inputs.extend(left_inputs)
            new_labels.extend(left_labels)
            new_groups.extend(left_groups)
            break

        else:
            new_inputs.extend(batch_inputs)
            new_labels.extend(batch_labels)
            new_groups.extend(batch_groups)

    new_inputs, new_labels, new_groups = np.array(new_inputs[::-1]), np.array(new_labels[::-1]), np.array(new_groups[::-1])
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(new_inputs), torch.from_numpy(new_labels), torch.from_numpy(new_groups))

    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

def custom_order_single_epoch(target_ratio, args):

    trainloader, validloader, testloader = get_dataset(dataset=args.dataset, protected_class=args.pc,
                                                       shuffle_seed=args.shuffle_seed, batch_size=args.batch_size)

    savefldr = get_experiment_fldr(args.ckptfldr, args.pc, args.loss, args.arch, args.init_seed, args.shuffle_seed)
    model = get_model(args.arch, args.num_feat, ckpt=savefldr + '99.pth', cuda=args.cuda, dropout=args.dropout)

    fscore, avg_odds = test_mlp(model, testloader, cuda=args.cuda, fairness_criteria='avgodds')
    print("Performance Before Custom Data Order -- F Score: %f; Avg Odds: %f" % (fscore, avg_odds))

    trainloader = make_trainloader_ratio(trainloader, target_ratio, args.batch_size)
    args.epochs = 1
    model = train_mlp(model, trainloader, validloader, None, args)

    fscore, avg_odds = test_mlp(model, testloader, cuda=args.cuda, fairness_criteria='avgodds')
    print("Performance After Custom Data Order -- F Score: %f; Avg Odds: %f" % (fscore, avg_odds))

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    ## Target Ratio for Custom Data Order
    ## target_ratio[(Group Index, Label Index)] = Required Ratio of The Subgroup
    # target_ratio = {(0, 0): 0.3, (0, 1): 0.3, (1, 0): 0.3, (1, 1): 0.1} # Adversarial Example
    target_ratio = {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.25} # Fair Example

    custom_order_single_epoch(target_ratio, args)
