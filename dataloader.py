import torch
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import Sampler

from folktables import ACSDataSource, ACSIncome, ACSEmployment
from fairbatch_local import FairBatch

def load_celeba_partition(img_list, celeba_feat_dir, ydict, groupdict):
    x, y, group = [], [], []
    for img in img_list:
        feat_path = os.path.join(celeba_feat_dir, img[:-3] + 'npy')
        if not os.path.exists(feat_path):
            continue

        feat = np.load(feat_path)
        x.append(feat)
        y.append(ydict[img])
        group.append(groupdict[img])

    return np.array(x), np.array(y), np.array(group)

def load_celeba_dataset():
    celeba_dir = '/mnt/LargeDisk/Data/celeba'
    celeba_label_file = os.path.join(celeba_dir, 'list_attr_celeba.csv')
    celeba_partition_file = os.path.join(celeba_dir, 'list_eval_partition.csv')
    celeba_feat_dir = os.path.join(celeba_dir, 'feat_align_celeba')

    dflabel = pd.read_csv(celeba_label_file)
    ydict = {img_id: smiling_label==1 for img_id, smiling_label in zip(dflabel['image_id'], dflabel['Smiling'])}
    groupdict = {img_id: 1-max(male_label, 0) for img_id, male_label in zip(dflabel['image_id'], dflabel['Male'])}

    dfpart = pd.read_csv(celeba_partition_file)
    img_list = dfpart['image_id']
    partition = dfpart['partition']
    train_img = img_list[partition==0]
    valid_img = img_list[partition==1]
    test_img = img_list[partition==2]

    x_train, y_train, group_train = load_celeba_partition(train_img, celeba_feat_dir, ydict, groupdict)
    x_valid, y_valid, group_valid = load_celeba_partition(valid_img, celeba_feat_dir, ydict, groupdict)
    x_test, y_test, group_test = load_celeba_partition(test_img, celeba_feat_dir, ydict, groupdict)

    return x_train, y_train, group_train, x_test, y_test, group_test, x_valid, y_valid, group_valid

def get_dataset(dataset='acsincome', protected_class='sex',
                shuffle_seed=0, batch_size=128, train_shuffle=True,
                fairbatch=False, model=None):

    if 'acs' in dataset:
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        acs_data = data_source.get_data(states=['CA'], download=True)

        if dataset=='acsincome':
            task_class = ACSIncome
        elif dataset=='acsemployment':
            task_class = ACSEmployment

        if protected_class=='sex':
            task_class._group = 'SEX'
            features, label, group = task_class.df_to_numpy(acs_data)
            group = group - 1
        elif protected_class=='race':
            task_class._group = 'RAC1P'
            features, label, group = task_class.df_to_numpy(acs_data)
            group[group>1] = 2 # White vs Others
            group = group - 1

        x_train, x_test, y_train, y_test, group_train, group_test = train_test_split(features, label, group, test_size=0.2, random_state=0) # Test Split 20%
        x_train, x_valid, y_train, y_valid, group_train, group_valid = train_test_split(x_train, y_train, group_train, test_size=0.1/0.8, random_state=0) # Val Split 10%

    elif dataset=='celeba':
        x_train, y_train, group_train, x_test, y_test, group_test, x_valid, y_valid, group_valid = load_celeba_dataset()

    ## Shuffle Training Data
    x_train, y_train, group_train = shuffle(x_train, y_train, group_train, random_state=shuffle_seed)

    datascaler = MinMaxScaler()
    datascaler.fit(x_train)
    x_train, x_valid, x_test = datascaler.transform(x_train), datascaler.transform(x_valid), datascaler.transform(x_test)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(group_train))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid), torch.from_numpy(group_valid))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(group_test))

    if fairbatch:
        tensorx_train, tensory_train, tensorgroup_train = torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(group_train)
        sampler = FairBatch(model, tensorx_train.cuda().float(), tensory_train.cuda().long(), tensorgroup_train.cuda(), batch_size=128,
                            alpha=0.005, target_fairness='eqodds', replacement=False, seed=0)
        trainloader = DataLoader(train_dataset, sampler=sampler)
    else:
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, drop_last=False)

    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return trainloader, validloader, testloader
