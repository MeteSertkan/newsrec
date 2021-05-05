import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention.additive_attention import AdditiveAttention


class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.mh_selfattention = nn.MultiheadAttention(
            config.word_embedding_dim,
            config.num_attention_heads)
        self.additive_attention = AdditiveAttention(config.query_vector_dim, config.word_embedding_dim)

    def forward(self, clicked_news_vector):
        # multi-head self-attention
        user_vector = clicked_news_vector.permute(1, 0, 2)
        user_vector, _ = self.mh_selfattention(
            user_vector,
            user_vector,
            user_vector)

        # additive-attention
        user_vector = user_vector.permute(1, 0, 2)
        # dropout here ? // there is also no in msr cod
        #user_vector = F.dropout(user_vector,
        #                                   p=self.config.dropout_probability,
        #                                   training=self.training)
        user_vector = self.additive_attention(user_vector)
        
        return user_vector
