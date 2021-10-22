import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import AdditiveAttention
from models.utils import TimeDistributed


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

class CategoryEncoder(torch.nn.Module):
    def __init__(self, category_embedding, linear_input_dim, linear_output_dim):
        super(CategoryEncoder, self).__init__()
        self.category_embedding = category_embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, element):
        return F.relu(self.linear(self.category_embedding(element)))

class NewsEncoder(nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        word_embedding = nn.Embedding.from_pretrained(
            pretrained_word_embedding,
            freeze=config.freeze_word_embeddings,
            padding_idx=0)
        category_embedding = nn.Embedding(
            config.num_categories,
            config.category_embedding_dim,
            padding_idx=0)
        self.title_encoder = TimeDistributed(
            TextEncoder(
                word_embedding, 
                config.word_embedding_dim,
                config.num_filters,
                config.window_size,
                config.query_vector_dim,
                config.dropout_probability),
            batch_first=True)
        self.abstract_encoder = TimeDistributed(
            TextEncoder(
                word_embedding, 
                config.word_embedding_dim,
                config.num_filters,
                config.window_size,
                config.query_vector_dim,
                config.dropout_probability),
            batch_first=True)
        self.category_encoder = CategoryEncoder(
            category_embedding,
            config.category_embedding_dim,
            config.num_filters)
        self.sub_category_encoder = CategoryEncoder(
            category_embedding,
            config.category_embedding_dim,
            config.num_filters)
        self.final_attention = AdditiveAttention(
            config.query_vector_dim,
            config.num_filters)

    def forward(self, title, abstract, category, subcategory):
        # text encoding
        title_vector = self.title_encoder(title)
        # abstract encoding
        abstract_vector = self.abstract_encoder(abstract)
        # category encoding
        category_vector = self.category_encoder(category)
        # subcategory encoding
        subcategory_vector = self.sub_category_encoder(subcategory)
        # combine vectors -> 4 vectors per news
        all_vectors = torch.stack(
            [title_vector,
            abstract_vector,
            category_vector,
            subcategory_vector],
            dim=2
        )
        # squash batch and sample (n-negative+1) axis
        all_vectors = all_vectors.contiguous().view(-1, all_vectors.size(-2), all_vectors.size(-1))
        final_news_vector = self.final_attention(
                all_vectors)
        # split back to batch x sample x encoding dim
        final_news_vector = final_news_vector.contiguous().view(-1, title.size(-2), final_news_vector.size(-1))
        return final_news_vector
