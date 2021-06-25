import torch
import torch.nn as nn
from torch.autograd import Variable
from models.sharedRep import SharedLayer


class TWGCNModel(nn.Module):
    def __init__(self, args, config, state_dict):
        super().__init__()
        self.args = args
        self.sharedLayer = SharedLayer(self.args, config, state_dict)
        in_dim = self.args.wgcn_hidden_dim
        self.fc1 = nn.Linear(in_dim, self.args.num_labels)
        

    def forward(self, input_ids, head_ids, segment_ids, input_mask, dep_ids):
        shared_features = self.sharedLayer(input_ids, head_ids, segment_ids, input_mask, dep_ids)
        level1 = self.fc1(shared_features)
        return level1
