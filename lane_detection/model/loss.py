import torch
import torch.nn as nn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ALPHA = 0.7
BETA = 0.3
GAMMA = 3

# iou loss
class IoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        mask = (targets != self.ignore_index).float()
        targets = targets.float()

        # area of overlap
        num = torch.sum(outputs*targets*mask)

        # area of union
        den = torch.sum(outputs*mask + targets*mask - outputs*targets*mask)
        return 1 - num/den

# ftl loss
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=.2, beta=.8, gamma=3):
        
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        # tversky index
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

        # ftl
        FocalTversky = (1 - Tversky)**gamma   
        return FocalTversky

