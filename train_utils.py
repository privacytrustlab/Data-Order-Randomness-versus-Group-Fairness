import time
import torch
import numpy as np
import torch.optim as optim
import torchbnn as bnn
from tqdm import tqdm
from sklearn.metrics import f1_score

from meta_fairness import equal_opp_binary, fair_loss_binary, avg_odds_binary, acc_diff_binary, disparate_impact_binary
from fairtorch_local import DemographicParityLoss, EqualiedOddsLoss

## Weights set manually using data distribution for Reweighing
## Make Sure to Change These Weights If Testing on a Different Dataset
def get_weights(labels, groups, dataset):
    if dataset=='acsincome': # ACSIncome; Protected Class: Sex
        reweigh_arr = [1.103221, 0.881572, 0.905575, 1.176071]
    elif dataset=='acsemployment': # ACSEmployment; Protected Class: Sex
        reweigh_arr = [1.066670, 0.930995, 0.943020, 1.077683]
    elif dataset=='celeba': # CelebA
        reweigh_arr = [0.866271, 1.201112, 1.125507, 0.892101]

    weights = torch.ones(labels.size())
    weights[torch.logical_and(labels==0, groups==0)] = reweigh_arr[0]
    weights[torch.logical_and(labels==1, groups==0)] = reweigh_arr[1]
    weights[torch.logical_and(labels==0, groups==1)] = reweigh_arr[2]
    weights[torch.logical_and(labels==1, groups==1)] = reweigh_arr[3]

    return weights

def train_mlp(model, trainloader, validloader, savefldr, args):

    torch.manual_seed(0)
    save_every_ite = True

    if args.reweigh:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss()
    dp_loss = EqualiedOddsLoss(sensitive_classes=[0, 1], alpha=args.lmbd)
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.01
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in tqdm(range(args.epochs)):
        if save_every_ite and savefldr is not None:
            torch.save(model, savefldr + '%d.pth' % epoch)

        model.train()
        torch.manual_seed(epoch)

        for i, data in enumerate(trainloader, 0):
            inputs, labels, groups = data
            inputs, labels, groups = inputs.squeeze(), labels.squeeze(), groups.squeeze()
            if args.cuda:
                inputs, labels, groups = inputs.cuda(), labels.cuda(), groups.cuda()

            optimizer.zero_grad()
            outputs = model(inputs.float())
            if args.loss=='ce':
                loss = criterion(outputs, labels.long())

                if args.reweigh:
                    weights = get_weights(labels, groups, args.dataset)
                    if args.cuda: weights = weights.cuda()
                    loss = torch.sum(loss * weights)/torch.sum(weights)
                if 'bnn' in args.arch:
                    kl = kl_loss(model)
                    loss = loss + kl_weight*kl

            elif args.loss=='fairce':
                lossce = criterion(outputs, labels.long())

                outprob = torch.nn.functional.softmax(outputs, dim=-1)[:, 1]
                lossfair = dp_loss(inputs, outprob, groups, labels.long())
                if torch.isnan(lossfair): loss = lossce
                else: loss = lossce + lossfair

            loss.backward()
            optimizer.step()

        # fscore, unfair_per = test_mlp(model, validloader, cuda=args.cuda)
        ## Use validloader results if required
    if savefldr is not None:
        torch.save(model, savefldr + 'final.pth')
    torch.manual_seed(int(time.time())) ## Reset Seed. This is important when training multiple models in succession

    return model

def test_mlp(model, testloader, cuda=True, fairness_criteria='eqopp'):

    model.eval()

    label_arr, pred_arr, group_arr = [], [], []
    with torch.no_grad():
        for data in testloader:
            inputs, labels, groups = data
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)

            pred_arr.extend(predicted.cpu().detach().numpy())
            label_arr.extend(labels.cpu().detach().numpy())
            group_arr.extend(groups.cpu().detach().numpy())

    label_arr = np.array(label_arr)
    pred_arr = np.array(pred_arr)
    group_arr = np.array(group_arr)

    fscore = f1_score(label_arr, pred_arr, average='macro')

    if fairness_criteria=='eqopp':
        unfair_per = equal_opp_binary(group_arr, label_arr, pred_arr)
    elif fairness_criteria=='avgodds':
        unfair_per = avg_odds_binary(group_arr, label_arr, pred_arr)
    elif fairness_criteria=='accdiff':
        unfair_per = acc_diff_binary(group_arr, label_arr, pred_arr)
    elif fairness_criteria=='disimp':
        unfair_per = disparate_impact_binary(group_arr, label_arr, pred_arr)

    return fscore*100, unfair_per*100
