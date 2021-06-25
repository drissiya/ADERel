import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def normalize_adj(A):
    I = np.eye(A.shape[0])
    A_hat = A + I 
    D_hat_diag = np.sum(A_hat, axis=1)
    D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
    D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
    D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
    return np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
	
def graph_to_adj(head_ids, input_mask, dep_ids, embedding_dep, sent_len, dim_emb, gcn_type):
    A = np.zeros((sent_len, sent_len), dtype=np.float32)
    m = len([i for i in input_mask if i==1])
    dep = dep_ids[:m]
    head = head_ids[:m]
    for idx, h in enumerate(head):
        if idx!=h:
            if gcn_type=='weighted':
                A[h,idx] = np.dot(embedding_dep[dep[h]],embedding_dep[dep[idx]])/dim_emb
            else:
                A[h,idx] = 1
    A = A + A.T
    A = normalize_adj(A)
    return A

class WGCN(nn.Module):
    def __init__(self, args, embeddings, mem_dim, num_layers):
        super(WGCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = embeddings.hidden_bert

        self.gcn_drop = nn.Dropout(self.args.dropout)

        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def forward(self, adj, gcn_inputs):
        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) 

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) 
        return gcn_inputs
