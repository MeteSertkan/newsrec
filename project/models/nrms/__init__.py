import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from models.nrms.news_encoder import NewsEncoder
from models.nrms.user_encoder import UserEncoder
from models.click_probability import DotProductClickPredictor
from models.metrics import NDCG, MRR, AUC, SentiMRR, Senti


class NRMS(pl.LightningModule):
    """
    NRMS network.
    Input 1 + K candidate news and a list of user clicked news,
    produce the click probability.
    """

    def __init__(self, config=None, pretrained_word_embedding=None):
        super(NRMS, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()
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

    def forward(self, candidate_news, clicked_news):
        # compute candidate news representation
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1)
        # compute browsed news representation
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        # compute user representation
        user_vector = self.user_encoder(clicked_news_vector)
        # compute click score
        click_probability = self.click_predictor(candidate_news_vector,
                                                 user_vector)
        return click_probability

    def training_step(self, batch, batch_idx):
        y_pred = self(batch["candidate_news"], batch["clicked_news"])
        y = torch.zeros(len(y_pred)).long().to(self.device)
        loss = F.cross_entropy(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred = self(batch["candidate_news"], batch["clicked_news"])
        y = batch["clicked"]
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
        y_pred = self(batch["candidate_news"], batch["clicked_news"])
        y = batch["clicked"]
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
        s_c_vader = []
        s_c_bert = []
        for x in batch["candidate_news"]:
            s_c_vader.append(x['vader_sentiment'])
            s_c_bert.append(x['distillbert_sst2_sentiment'])
        s_c_vader = torch.stack(s_c_vader,dim=1).flatten()
        s_c_bert = torch.stack(s_c_bert,dim=1).flatten()

        # calc mean sentiment score from browsed news
        # (using sentiment classifier)
        s_clicked_vader = []
        s_clicked_bert = []
        for x in batch["clicked_news"]:
            s_clicked_vader.append(x['vader_sentiment'])
            s_clicked_bert.append(x['distillbert_sst2_sentiment'])
        s_clicked_vader = torch.stack(s_clicked_vader, dim=1).flatten()
        s_clicked_bert = torch.stack(s_clicked_bert, dim=1).flatten()
        s_mean_vader = s_clicked_vader.mean()
        s_mean_bert = s_clicked_bert.mean()

        return s_c_vader, s_c_bert, s_mean_vader, s_mean_bert

    def get_news_vector(self, news):
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
