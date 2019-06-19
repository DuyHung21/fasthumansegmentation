from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainMode(IntEnum):
    SEG = 1
    REFINE = 2
    SEG_REFINE = 3

def seg_matting_loss(img, pred_seg, gt_seg, pred_alpha, gt_alpha, mode=TrainMode.SEG): 
    criterion = nn.CrossEntropyLoss()
    cross_entropy_loss = criterion(pred_seg, gt_seg[:, 1, :, :].long())
    if mode == TrainMode.SEG:
        return cross_entropy_loss

    esp = 1e-6
    L_alpha = torch.sqrt(torch.pow(gt_alpha - pred_alpha, 2.) + esp).mean()

    gt_alpha_img = torch.cat((gt_alpha, gt_alpha, gt_alpha), 1) * img
    pred_alpha_img = torch.cat((pred_alpha, pred_alpha, pred_alpha), 1) * img
    L_color = torch.sqrt(torch.pow(gt_alpha_img - pred_alpha_img, 2.) + esp).mean()

    if mode == TrainMode.REFINE:
        return L_alpha + L_color
    else:
        return L_alpha + L_color + cross_entropy_loss
