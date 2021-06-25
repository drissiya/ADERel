import torch
import torch.nn as nn

class JointLossWrapper(nn.Module):
    def __init__(self, level_num, device):
        super(JointLossWrapper, self).__init__()
        self.level_num = level_num
        self.device = device
        self.log_vars = nn.Parameter(torch.zeros((level_num)))
        self.loss_fn1 = nn.CrossEntropyLoss().to(self.device)
        self.loss_fn2 = nn.CrossEntropyLoss().to(self.device)

    def forward(self, preds, levels):
        loss1 = self.loss_fn1(preds[0],levels[0])
        loss2 = self.loss_fn2(preds[1],levels[1])

        loss1 = torch.exp(-self.log_vars[0])*loss1 + self.log_vars[0]
        loss2 = torch.exp(-self.log_vars[1])*loss2 + self.log_vars[0]
        
        return loss1+loss2