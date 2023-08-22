import torch
import numpy as np
import os

import torchbnn as bnn

arch_id_to_hidden_layers = {'mlp_16':[16], 'mlp_64':[64], 'mlp_64_bnn': [64],
                            'mlp_2048_64':[2048, 64], 'mlp_64_64_128':[64, 64, 128]}

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_arr=[], dropout=0., bnn_layers=False):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.relu = torch.nn.ReLU()
        if dropout!=0.:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        prev = self.input_size
        self.fca = []
        for ele in hidden_arr:
            if bnn_layers:
                self.fca.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=prev, out_features=ele))
            else:
                self.fca.append(torch.nn.Linear(prev, ele))
            prev = ele
        self.fca = torch.nn.ModuleList(self.fca)

        if bnn_layers:
            self.fcout = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=prev, out_features=2)
        else:
            self.fcout = torch.nn.Linear(prev, 2)

    def forward(self, x):
        for fcele in self.fca:
            x = fcele(x)
            try:
                x = self.dropout(x)
            except:
                pass
            x = self.relu(x)

        out = self.fcout(x)
        return out

def get_model(arch_id, input_size, ckpt=None, cuda=True, dropout=0.):
    hidden_layers = arch_id_to_hidden_layers[arch_id]

    if ckpt is not None:
        model = torch.load(ckpt)
    elif 'bnn' in arch_id:
        model = Feedforward(input_size, hidden_layers, dropout=dropout, bnn_layers=True)
    else:
        model = Feedforward(input_size, hidden_layers, dropout=dropout)

    if cuda: model = model.cuda()
    return model

def get_experiment_fldr(ckptfldr, protected_class, losstype, arch_id, init_seed, shuffle_seed):
    experiment_fldr = os.path.join("models", ckptfldr, protected_class, losstype, "%s_%d_%d" % (arch_id, init_seed, shuffle_seed), '')
    os.makedirs(os.path.dirname(experiment_fldr), exist_ok=True)

    return experiment_fldr

########## Loading a standard model for Monte Carlo Dropout Inference #########
class FeedforwardMCD(torch.nn.Module):
    def __init__(self, model, dropout=0.1):
        super(FeedforwardMCD, self).__init__()
        self.model = model
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        for fcele in self.model.fca:
            x = fcele(x)
            x = self.dropout(x)
            x = self.model.relu(x)

        out = self.model.fcout(x)
        return out
