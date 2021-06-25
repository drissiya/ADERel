import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class BertEncoder(nn.Module):
    def __init__(self, args, config, state_dict):
        super().__init__()
        self.bert = BertModel(config)
        self.bert.load_state_dict(state_dict, strict=False)
        self.args = args
        self.hidden_bert = self.bert.config.hidden_size
        self.dropout_bert = self.bert.config.hidden_dropout_prob
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, input_ids, token_type_ids, attention_mask):
        encoded_layer, _ = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        encl = encoded_layer[-1]
        sequence_output = self.dropout(encl)
        return sequence_output