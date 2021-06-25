import torch
import math
import torch.nn as nn

class ScaledDotProduct(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProduct, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        d_k = q.size(-1)
        attention = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d_k)

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProduct(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query):
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)
        entity_size = query.size(1)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.reshape(batch_size, -1, num_heads, dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)
        value = value.reshape(batch_size, -1, num_heads, dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)
        query = query.reshape(batch_size, -1, num_heads,  dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)

        context, attention = self.dot_product_attention(
            query, key, value)

        # concat heads
        context = context.reshape(batch_size,num_heads, -1,  dim_per_head).transpose(1,2).reshape(batch_size, -1, num_heads*dim_per_head)

        output = self.linear_final(context)

        output = self.dropout(output)

        return output
		
class SelfAttention(nn.Module):

    def __init__(self, dimensions, attention_type='general'):
        super(Attention1, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = torch.bmm(attention_weights, context)


        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)


        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output
