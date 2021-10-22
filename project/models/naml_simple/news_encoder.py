import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import AdditiveAttention


class TextEncoder(torch.nn.Module):
    def __init__(self, word_embedding, word_embedding_dim, num_filters,
                 window_size, query_vector_dim, dropout_probability):
        super(TextEncoder, self).__init__()
        self.dropout_probability = dropout_probability
        self.word_embedding = word_embedding
        assert window_size >= 1 and window_size % 2 == 1
        self.CNN = nn.Conv1d(
            1,
            num_filters,
            (window_size, word_embedding_dim),
            padding=(int((window_size - 1) / 2), 0))
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)

    def forward(self, text):
        # word embedding
        text_vector = F.dropout(
            self.word_embedding(text),
            p=self.dropout_probability,
            training=self.training)
        # CNN contextualization
        text_vector = self.CNN(
            text_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # dropout
        text_vector = F.dropout(
            F.relu(text_vector),
            p=self.dropout_probability,
            training=self.training)
        # additive-attention
        text_vector = self.additive_attention(
            text_vector.transpose(1, 2))

        return text_vector

class NewsEncoder(nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        word_embedding = nn.Embedding.from_pretrained(
            pretrained_word_embedding,
            freeze=config.freeze_word_embeddings,
            padding_idx=0)
        self.title_encoder = TextEncoder(
                word_embedding, 
                config.word_embedding_dim,
                config.num_filters,
                config.window_size,
                config.query_vector_dim,
                config.dropout_probability)

    def forward(self, title):
        # text encoding
        title_vector = self.title_encoder(title)
        return title_vector
