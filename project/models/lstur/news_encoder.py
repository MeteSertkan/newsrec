import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import AdditiveAttention
from models.utils import TimeDistributed


class NewsEncoder(nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding.from_pretrained(
            pretrained_word_embedding, 
            freeze=config.freeze_word_embeddings, 
            padding_idx=0)
        assert config.window_size >= 1 and config.window_size % 2 == 1
        self.title_CNN =nn.Conv1d(
            1,
            config.num_filters,
            (config.window_size, config.word_embedding_dim),
            padding=(int((config.window_size - 1) / 2), 0))
        self.title_attention = AdditiveAttention(
            config.query_vector_dim, 
            config.num_filters)

    def forward(self, title):
        # word embedding
        title_vector = F.dropout(self.word_embedding(
            title),
            p=self.config.dropout_probability,
            training=self.training)
        # squash batch and sample (n-negative+1) axis
        title_vector = title_vector.contiguous().view(-1, 1, title_vector.size(-2), title_vector.size(-1))
        # CNN contextualization
        title_vector = self.title_CNN(
            title_vector).squeeze(dim=3)
        title_vector = F.dropout(
            F.relu(title_vector),
            p=self.config.dropout_probability,
            training=self.training)
        # attention
        title_vector = self.title_attention(
            title_vector.transpose(1, 2))
        # split back title vector to batch x 1+N-negative x Embedding-Dim
        title_vector = title_vector.contiguous().view(-1, title.size(-2), title_vector.size(-1))
        return title_vector
