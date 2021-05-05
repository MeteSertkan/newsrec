from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
import torch


class BaseDataset(Dataset):
    def __init__(self,
                 behaviors_path,
                 news_path,
                 config):
        super(BaseDataset, self).__init__()
        self.config = config
        assert all(attribute in [
            'category', 'subcategory', 'title', 'abstract', 'title_entities',
            'abstract_entities', 'vader_sentiment', 'distillbert_sst2_sentiment'
        ] for attribute in config.dataset_attributes.news)
        assert all(attribute in ['user', 'clicked_news_length']
                   for attribute in config.dataset_attributes.record)

        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            usecols=['id'] + config.dataset_attributes.news,
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes.news) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities',
                    'vader_sentiment', 'distillbert_sst2_sentiment'
                ])
            })
        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
        self.news2dict = self.news_parsed.to_dict('index')
        padding_all = {
            'category': 0,
            'subcategory': 0,
            'title': [0] * config.num_words_title,
            'abstract': [0] * config.num_words_abstract,
            'title_entities': [0] * config.num_words_title,
            'abstract_entities': [0] * config.num_words_abstract,
            'vader_sentiment': 0.0, 
            'distillbert_sst2_sentiment': 0.0
        }
        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        self.padding = {
            k: v
            for k, v in padding_all.items()
            if k in config.dataset_attributes.news
        }

    def _news2dict(self, id):
        ret = self.news2dict[id]
        for key in ret.keys():
            if torch.is_tensor(ret[key]):
                ret[key] = ret[key]
            else:
                ret[key] = torch.tensor(ret[key])

        return ret

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in self.config.dataset_attributes.record:
            item['user'] = torch.tensor(row.user)
        item["clicked"] = torch.tensor(list(map(int, row.clicked.split())))
        item["candidate_news"] = [
            self._news2dict(x) for x in row.candidate_news.split()
        ]
        item["clicked_news"] = [
            self._news2dict(x)
            for x in row.clicked_news.split()[:self.config.num_clicked_news_a_user]   # noqa: E501
        ]
        if 'clicked_news_length' in self.config.dataset_attributes['record']:
            item['clicked_news_length'] = torch.tensor(len(item["clicked_news"]))
        repeated_times = self.config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding
                                ] * repeated_times + item["clicked_news"]
        return item
