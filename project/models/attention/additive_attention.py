import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, embedding_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(embedding_dim, query_dim)
        self.query = nn.Parameter(
            torch.empty((query_dim, 1)).uniform_(-0.1, 0.1))

    def forward(self, input_sequence):
        attention = torch.tanh(self.linear(input_sequence))
        attention = torch.matmul(attention, self.query).squeeze(dim=-1)
        attention_weights = F.softmax(attention,
                                      dim=1)
        weighted_input = torch.bmm(attention_weights.unsqueeze(dim=1),
                                   input_sequence).squeeze(dim=1)
        return weighted_input
