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


# credits to miguelvr
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y