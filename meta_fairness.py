import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import torch
import torch.nn.functional as F
from fairtorch import ConstraintLoss

def equal_opp_binary(sensitive_class, label_arr, pred_arr, return_class_scores=False):

    indicesz = np.argwhere(sensitive_class==0)
    labelz = label_arr[indicesz]
    predz = pred_arr[indicesz]

    confz = confusion_matrix(labelz, predz)
    tprz = confz[1][1]/(confz[1][1] + confz[1][0])
    prec_z = confz[1][1]/(confz[1][1] + confz[0][1])

    indiceso = np.argwhere(sensitive_class==1)
    labelo = label_arr[indiceso]
    predo = pred_arr[indiceso]

    confo = confusion_matrix(labelo, predo)
    tpro = confo[1][1]/(confo[1][1] + confo[1][0])
    prec_o = confo[1][1]/(confo[1][1] + confo[0][1])

    if return_class_scores:
        return abs(tprz - tpro), {'prec_z': prec_z, 'rec_z': tprz, 'prec_o': prec_o, 'rec_o': tpro}
    else:
        return abs(tprz - tpro)


def avg_odds_binary(sensitive_class, label_arr, pred_arr, return_class_scores=False):

    indicesz = np.argwhere(sensitive_class==0)
    labelz = label_arr[indicesz]
    predz = pred_arr[indicesz]

    # print(np.unique(labelz, return_counts=True))
    # print(np.unique(predz, return_counts=True))
    # print(predz)
    # print(labelz.shape)
    # print(predz.shape)

    confz = confusion_matrix(labelz, predz)
    tprz = confz[1][1]/(confz[1][1] + confz[1][0])
    fprz = confz[0][1]/(confz[0][0] + confz[0][1])
    prec_z = confz[1][1]/(confz[1][1] + confz[0][1])

    indiceso = np.argwhere(sensitive_class==1)
    labelo = label_arr[indiceso]
    predo = pred_arr[indiceso]

    confo = confusion_matrix(labelo, predo)
    tpro = confo[1][1]/(confo[1][1] + confo[1][0])
    fpro = confo[0][1]/(confo[0][0] + confo[0][1])
    prec_o = confo[1][1]/(confo[1][1] + confo[0][1])


    if return_class_scores:
        return (abs(tprz - tpro) + abs(fprz - fpro))/2, {'prec_z': prec_z, 'rec_z': tprz, 'prec_o': prec_o, 'rec_o': tpro}
    else:
        # return (tprz - tpro + fprz - fpro)/2
        return (abs(tprz - tpro) + abs(fprz - fpro))/2

def acc_diff_binary(sensitive_class, label_arr, pred_arr, return_class_scores=False):

    indicesz = np.argwhere(sensitive_class==0)
    labelz = label_arr[indicesz]
    predz = pred_arr[indicesz]

    accz = accuracy_score(labelz, predz)
    confz = confusion_matrix(labelz, predz)
    tprz = confz[1][1]/(confz[1][1] + confz[1][0])
    fprz = confz[0][1]/(confz[0][0] + confz[0][1])
    prec_z = confz[1][1]/(confz[1][1] + confz[0][1])

    indiceso = np.argwhere(sensitive_class==1)
    labelo = label_arr[indiceso]
    predo = pred_arr[indiceso]

    acco = accuracy_score(labelo, predo)
    confo = confusion_matrix(labelo, predo)
    tpro = confo[1][1]/(confo[1][1] + confo[1][0])
    fpro = confo[0][1]/(confo[0][0] + confo[0][1])
    prec_o = confo[1][1]/(confo[1][1] + confo[0][1])

    if return_class_scores:
        return abs(accz - acco), {'prec_z': prec_z, 'rec_z': tprz, 'prec_o': prec_o, 'rec_o': tpro}
    else:
        return abs(accz - acco)

def disparate_impact_binary(sensitive_class, label_arr, pred_arr, return_class_scores=False):

    indicesz = np.argwhere(sensitive_class==0)
    labelz = label_arr[indicesz]
    predz = pred_arr[indicesz]

    diz = np.sum(predz==1)/len(predz)
    confz = confusion_matrix(labelz, predz)
    tprz = confz[1][1]/(confz[1][1] + confz[1][0])
    fprz = confz[0][1]/(confz[0][0] + confz[0][1])
    prec_z = confz[1][1]/(confz[1][1] + confz[0][1])
    # diz = (confz[1][1] + confz[0][1])/(confz[1][1] + confz[1][0] + confz[0][0] + confz[0][1])

    indiceso = np.argwhere(sensitive_class==1)
    labelo = label_arr[indiceso]
    predo = pred_arr[indiceso]

    dio = np.sum(predo==1)/len(predo)
    confo = confusion_matrix(labelo, predo)
    tpro = confo[1][1]/(confo[1][1] + confo[1][0])
    fpro = confo[0][1]/(confo[0][0] + confo[0][1])
    prec_o = confo[1][1]/(confo[1][1] + confo[0][1])
    # dio = (confo[1][1] + confo[0][1])/(confo[1][1] + confo[1][0] + confo[0][0] + confo[0][1])

    if return_class_scores:
        return 1 - min(dio/diz, diz/dio), {'prec_z': prec_z, 'rec_z': tprz, 'prec_o': prec_o, 'rec_o': tpro}
    else:
        return 1 - min(dio/diz, diz/dio)

def fair_loss_binary(output, labels, protected_class, lmbd=1):

    output_softmax = F.softmax(output, dim=1)

    zindex = torch.where((protected_class == 0) & (labels == 1))
    oindex = torch.where((protected_class == 1) & (labels == 1))

    tp_zero = torch.sum(output_softmax[zindex][:, 0])
    fn_zero = torch.sum(output_softmax[zindex][:, 1])
    tpr_zero = torch.true_divide(tp_zero, tp_zero + fn_zero + 1e-7)

    tp_one = torch.sum(output_softmax[oindex][:, 1])
    fn_one = torch.sum(output_softmax[oindex][:, 0])
    tpr_one = torch.true_divide(tp_one, tp_one + fn_one + 1e-7)

    fair_loss = torch.nn.L1Loss()(tpr_one, tpr_zero)

    return lmbd*fair_loss
