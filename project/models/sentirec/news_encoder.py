import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import AdditiveAttention


class NewsEncoder(nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding.from_pretrained(
            pretrained_word_embedding, 
            freeze=config.freeze_word_embeddings, 
            padding_idx=0)
        self.mh_selfattention = nn.MultiheadAttention(
            config.word_embedding_dim,
            config.num_attention_heads)
        self.additive_attention = AdditiveAttention(config.query_vector_dim, config.word_embedding_dim)

    def forward(self, title):
        # word embedding
        title_vector = F.dropout(self.word_embedding(
            title),
            p=self.config.dropout_probability,
            training=self.training)
        # multi-head self attention
        title_vector = title_vector.permute(1, 0, 2)
        title_vector, _ = self.mh_selfattention(
            title_vector,
            title_vector,
            title_vector)
        # additive attention
        title_vector = title_vector.permute(1, 0, 2)
        title_vector = F.dropout(title_vector,
                                           p=self.config.dropout_probability,
                                           training=self.training)
        title_vector = self.additive_attention(title_vector)

        return title_vector
