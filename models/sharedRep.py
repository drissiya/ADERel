import torch
import torch.nn as nn
from torch.autograd import Variable
from models.bert import BertEncoder
from models.wgcn import WGCN, graph_to_adj
import numpy as np



class SharedLayer(nn.Module):
    def __init__(self, args, config, state_dict):
        super().__init__()
        self.args = args

        #BERT layer
        self.sentence_encoder = BertEncoder(args= self.args, config = config, state_dict=state_dict)
            
        # WGCN layer
        if self.args.gcn_type !='None':
            self.wgcn = WGCN(self.args, self.sentence_encoder, args.wgcn_hidden_dim, args.wgcn_num_layers)
            

    def forward(self, input_ids, head_ids, segment_ids, input_mask, dep_ids):
        shared_rep = self.sentence_encoder(input_ids, segment_ids, input_mask)
        if self.args.gcn_type !='None':
            adj = [graph_to_adj(head_ids[i], input_mask[i], dep_ids[i], self.args.dep_matrix, self.args.max_seq_length, self.args.dim_emb, self.args.gcn_type).reshape(1, self.args.max_seq_length, self.args.max_seq_length) for i in range(input_mask.shape[0])]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            adj = adj.to(torch.float32)
            adj = Variable(adj.cuda()) if self.args.cuda else Variable(adj)

            shared_rep = self.wgcn(adj, shared_rep)
        return shared_rep
