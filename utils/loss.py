import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
import sys
class lossfunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.diceloss = BinaryDiceLoss()
        self.focalloss = WeightedFocalLoss(alpha = 0.01)
        self.alpha = 0.5
        #self.maxpool = nn.MaxPool2d()
    def forward(self, preds, targets):
        loss = 0
        for idx,pred in enumerate(preds):
            target = nn.AdaptiveMaxPool2d(80//(2**idx))(targets)
            loss_dice = self.diceloss(pred, target)
            loss_focal = self.focalloss(pred, target)
            loss += (self.alpha)*loss_dice + (1-self.alpha)*loss_focal
        return loss, loss_dice, loss_focal


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pred, targets):
        n,c,h,w = pred.shape
        assert(pred.shape == targets.shape)
        
        pred = self.softmax(pred)
        
        pred = pred[:,0,:,:]
        targets = targets[:,0,:,:]
        N = targets.size()[0]
        smooth = 1

        pred_flat = pred.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 交集
        intersection = pred_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss" 
    def __init__(self, alpha=.25, gamma=2):
        '''
        alpha是第一类（背景类）的系数
        '''
        super(WeightedFocalLoss, self).__init__()        
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()        
        self.gamma = gamma
            
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') 
        
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))        #获取真实标签   
        
        pt = torch.exp(-BCE_loss)
        
        F_loss = (1-pt)**self.gamma * BCE_loss
        F_loss = at*F_loss.view(-1)
        return F_loss.mean()