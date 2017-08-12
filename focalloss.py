import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        logpt = F.log_softmax(input)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.index_select(1,target).diag(0).mean()
        return loss