import torch
import torch.nn as nn
from torch.autograd import Variable
from models.sharedRep import SharedLayer
from models.ajtwgcn.attention import MultiHeadAttention, SelfAttention


class AJTWGCNModel(nn.Module):
    def __init__(self, args, config, state_dict):
        super().__init__()
        self.args = args
        self.sharedLayer = SharedLayer(self.args, config, state_dict)
        in_dim = self.args.wgcn_hidden_dim
        if self.args.attention_type == "multi-head":
            self.multiheadattention = MultiHeadAttention(model_dim=in_dim, num_heads=self.args.num_heads, dropout=self.args.dropout)
        elif self.args.attention_type == "self":
            self.selfattention = SelfAttention(in_dim)

        self.fc1 = nn.Linear(in_dim, self.args.num_labels)
        self.fc2 = nn.Linear(in_dim, self.args.num_labels)
        

    def forward(self, input_ids, head_ids, segment_ids, input_mask, dep_ids):
        shared_features = self.sharedLayer(input_ids, head_ids, segment_ids, input_mask, dep_ids)
        level1 = self.fc1(shared_features)
        if self.args.attention_type == "multi-head":
            shared_features = self.multiheadattention(shared_features, shared_features, shared_features)
        elif self.args.attention_type == "self":
            shared_features = self.selfattention(shared_features, shared_features)
        level2 = self.fc2(shared_features)

        return level1, level2
