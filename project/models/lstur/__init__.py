import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from models.lstur.news_encoder import NewsEncoder
from models.lstur.user_encoder import UserEncoder
from models.metrics import NDCG, MRR, AUC, SentiMRR, Senti


class LSTUR(pl.LightningModule):
    def __init__(self, config=None, pretrained_word_embedding=None):
        super(LSTUR, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        assert int(config.num_filters * 0.5) == config.num_filters * 0.5
        self.user_embedding = nn.Embedding(
            config.num_users,
            config.num_filters if config.long_short_term_method == 'ini'
            else int(config.num_filters * 0.5),
            padding_idx=0)
        # val metrics
        self.val_performance_metrics = MetricCollection({
            'val_auc': AUC(),
            'val_mrr': MRR(),
            'val_ndcg@5': NDCG(k=5),
            'val_ndcg@10': NDCG(k=10)
        })
        self.val_sentiment_diversity_metrics_vader = MetricCollection({
            'val_senti_mrr_vader': SentiMRR(),
            'val_senti@5_vader': Senti(k=5),
            'val_senti@10_vader': Senti(k=10)
        })
        self.val_sentiment_diversity_metrics_bert = MetricCollection({
            'val_senti_mrr_bert': SentiMRR(),
            'val_senti@5_bert': Senti(k=5),
            'val_senti@10_bert': Senti(k=10)
        })
        # test metrics
        self.test_performance_metrics = MetricCollection({
            'test_auc': AUC(),
            'test_mrr': MRR(),
            'test_ndcg@5': NDCG(k=5),
            'test_ndcg@10': NDCG(k=10)
        })
        self.test_sentiment_diversity_metrics_vader = MetricCollection({
            'test_senti_mrr_vader': SentiMRR(),
            'test_senti@5_vader': Senti(k=5),
            'test_senti@10_vader': Senti(k=10)
        })
        self.test_sentiment_diversity_metrics_bert = MetricCollection({
            'test_senti_mrr_bert': SentiMRR(),
            'test_senti@5_bert': Senti(k=5),
            'test_senti@10_bert': Senti(k=10)
        })

    def forward(self, batch):
        # encode candidate news
        candidate_news_vector = self.news_encoder(
            batch["c_title"]
            )
        # encode history 
        clicked_news_vector = self.news_encoder(
            batch["h_title"]
            )
        # compute embedding (long-term interests)
        user = F.dropout2d(
            self.user_embedding(batch["user"]).unsqueeze(dim=0),
            p=self.config.masking_probability,
            training=self.training).squeeze(dim=0)
        # compute user representation
        user_vector = self.user_encoder(user, batch["history_length"],
                                        clicked_news_vector)
        # compute scores for each candidate news
        clicks_score = torch.bmm(
            candidate_news_vector,
            user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        clicks_score = F.softmax(clicks_score, dim=1)
        
        return clicks_score

    def training_step(self, batch, batch_idx):
        y_pred = self(batch)
        #y_pred = torch.sigmoid(y_pred)
        y = torch.zeros(len(y_pred)).long().to(self.device)
        loss = F.cross_entropy(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred = self(batch)
        #y_pred = F.softmax(y_pred, dim=1)
        y = batch["labels"]
        # determine candidate sentiment and overall sentiment orientation
        s_c_vader, s_c_bert, s_mean_vader, s_mean_bert = self.sentiment_evaluation_helper(batch)
        # compute metrics
        self.val_performance_metrics(y_pred, y)
        self.val_sentiment_diversity_metrics_vader(y_pred.flatten(), s_c_vader, s_mean_vader)
        self.val_sentiment_diversity_metrics_bert(y_pred.flatten(), s_c_bert, s_mean_bert)
        # log metric
        self.log_dict(self.val_performance_metrics, on_step=True, on_epoch=True)
        self.log_dict(self.val_sentiment_diversity_metrics_vader, on_step=True, on_epoch=True)
        self.log_dict(self.val_sentiment_diversity_metrics_bert, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        y_pred = self(batch)
        #y_pred = F.softmax(y_pred, dim=1)
        y = batch["labels"]
        # determine candidate sentiment and overall sentiment orientation
        s_c_vader, s_c_bert, s_mean_vader, s_mean_bert = self.sentiment_evaluation_helper(batch)
        # compute metrics
        self.test_performance_metrics(y_pred, y)
        self.test_sentiment_diversity_metrics_vader(y_pred.flatten(), s_c_vader, s_mean_vader)
        self.test_sentiment_diversity_metrics_bert(y_pred.flatten(), s_c_bert, s_mean_bert)
        # log metric
        self.log_dict(self.test_performance_metrics, on_step=True, on_epoch=True)
        self.log_dict(self.test_sentiment_diversity_metrics_vader, on_step=True, on_epoch=True)
        self.log_dict(self.test_sentiment_diversity_metrics_bert, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.config.learning_rate)
    
    def sentiment_evaluation_helper(self, batch):
        # sentiment scores of candidate news
        # (determined through sentiment classifier)
        s_c_vader = batch["c_vader_sentiment"].flatten()
        s_c_bert = batch["c_bert_sentiment"].flatten()
        # calc mean sentiment score from browsed news
        # (using sentiment classifier
        s_clicked_vader = batch["h_vader_sentiment"].flatten()
        s_clicked_bert = batch["h_bert_sentiment"].flatten()
        s_mean_vader = s_clicked_vader.mean()
        s_mean_bert = s_clicked_bert.mean()

        return s_c_vader, s_c_bert, s_mean_vader, s_mean_bert
