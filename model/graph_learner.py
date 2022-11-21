"""
Adapted from https://github.com/hugochan/IDGL/blob/master/src/core/layers/graphlearn.py
Author: hugochan
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VERY_SMALL_NUMBER = 1e-12
INF = 1e20



class GraphLearner(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_nodes,
        num_heads=1,
        embed_dim=10,
        metric_type="self_attention",
    ):
        super(GraphLearner, self).__init__()

        self.num_nodes = num_nodes
        self.metric_type = metric_type

        if metric_type == "weighted_cosine":
            self.weight_tensor = torch.Tensor(num_heads, input_size)
            self.weight_tensor = nn.Parameter(
                nn.init.xavier_uniform_(self.weight_tensor)
            )

        elif metric_type == "self_attention":
            self.linear_Q = nn.Linear(input_size, hidden_size, bias=False)
            self.linear_K = nn.Linear(input_size, hidden_size, bias=False)

        elif metric_type == "cosine": # no learnable params
            pass

        elif metric_type == "adaptive":
            # for adaptive GSL, each "head" is for one graph within a temporal resolution
            if num_heads > 1:
                self.E1 = torch.FloatTensor(num_heads, num_nodes, embed_dim)
            else:
                self.E1 = torch.FloatTensor(num_nodes, embed_dim)
            self.E1 = nn.Parameter(
                nn.init.xavier_uniform_(self.E1)
            )

        else:
            raise ValueError("Unknown metric_type: {}".format(metric_type))

    def forward(self, context, attn_mask=None, batch_size=None):
        """
        Args:
            context: (batch, num_nodes, dim)
            attn_mask: (batch, num_nodes, num_nodes), 0 will be masked out as 0 in attention
        Returns:
            attention: (batch, num_nodes, num_nodes)
        """
        if self.metric_type == "weighted_cosine":
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(
                0
            )
            attention[attention < 0] = 0

            # optional masking
            markoff_value = 0
            if attn_mask is not None:
                if len(attn_mask.shape) == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                attention = attention.masked_fill_(
                    torch.logical_not(attn_mask.bool()), markoff_value
                )

        elif self.metric_type == "self_attention":
            Q = self.linear_Q(context)
            K = self.linear_K(context)

            attention = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])

            # optional masking
            markoff_value = -INF
            if attn_mask is not None:
                if len(attn_mask.shape) == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                attention = attention.masked_fill_(
                    torch.logical_not(attn_mask.bool()), markoff_value
                )

            attention = torch.softmax(attention, dim=-1)

        elif self.metric_type == "cosine":
            context_norm = F.normalize(context.unsqueeze(0), p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(
                0
            )
            attention[attention < 0] = 0

            # optional masking
            markoff_value = 0
            if attn_mask is not None:
                if len(attn_mask.shape) == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                attention = attention.masked_fill_(
                    torch.logical_not(attn_mask.bool()), markoff_value
                )

        elif self.metric_type == "adaptive":
            attention = F.leaky_relu(torch.matmul(self.E1, self.E1.transpose(-1,-2)))

            # repeat by batch size
            if len(self.E1.shape) == 3:
                attention = attention.repeat(batch_size, 1, 1, 1)
            else:
                attention = attention.repeat(batch_size, 1, 1)

            # optional masking
            markoff_value = -INF
            if attn_mask is not None:
                if len(attn_mask.shape) == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                attention = attention.masked_fill_(
                    torch.logical_not(attn_mask.bool()), markoff_value
                )

            attention = torch.softmax(attention, dim=-1).reshape(-1, self.num_nodes, self.num_nodes)

        else:
            raise NotImplementedError()
        
        return attention