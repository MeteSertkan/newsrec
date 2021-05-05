import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention.additive_attention import AdditiveAttention


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
        if pretrained_word_embedding is None:
            word_embedding = nn.Embedding(
                config.num_words,
                config.word_embedding_dim,
                padding_idx=0)
        else:
            word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding,
                freeze=config.freeze_word_embeddings,
                padding_idx=0)
        category_embedding = nn.Embedding(config.num_categories,
                                          config.category_embedding_dim,
                                          padding_idx=0)
       
        # text encoders with shared word embedding layer
        assert len(config.dataset_attributes['news']) > 0
        text_encoders_candidates = ['title', 'abstract']
        self.text_encoders = nn.ModuleDict({
            name:
            TextEncoder(word_embedding, config.word_embedding_dim,
                        config.num_filters, config.window_size,
                        config.query_vector_dim, config.dropout_probability)
            for name in (set(config.dataset_attributes['news'])
                            & set(text_encoders_candidates))
        })

        # category encoders with shared category embeding layer
        element_encoders_candidates = ['category', 'subcategory']
        self.category_encoders = nn.ModuleDict({
            name:
            CategoryEncoder(category_embedding, config.category_embedding_dim,
                           config.num_filters)
            for name in (set(config.dataset_attributes['news'])
                         & set(element_encoders_candidates))
        })
        if len(config.dataset_attributes['news']) > 1:
            self.final_attention = AdditiveAttention(config.query_vector_dim,
                                                     config.num_filters)

    def forward(self, news):
        # text encoding - title, abstarct... 
        text_vectors = [
            encoder(news[name])
            for name, encoder in self.text_encoders.items()
        ]
        # category encoding - cat, subcat... 
        category_vectors = [
            encoder(news[name])
            for name, encoder in self.category_encoders.items()
        ]
        # additive-attention over all vectors
        all_vectors = text_vectors + category_vectors
        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            final_news_vector = self.final_attention(
                torch.stack(all_vectors, dim=1))
        return final_news_vector
