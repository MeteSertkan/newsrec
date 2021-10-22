from torch.utils.data import Dataset
import torch
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(self, behavior_path, news_path, config):
        super(BaseDataset, self).__init__()
        self.config = config
        self.behaviors_parsed = []
        news_parsed = {}
        #
        # loading and preparing news collection
        #       
        with open(news_path, 'r') as file:
            news_collection = file.readlines()
            for news in tqdm(news_collection):
                nid, cat, subcat, title, abstract, vader_sent, bert_sent = news.split("\t")
                news_parsed[nid] = {
                    'category': torch.tensor(int(cat)),
                    'subcategory': torch.tensor((int(subcat))),
                    'title': torch.tensor([int(i) for i in title.split(" ")]), 
                    'abstract': torch.tensor([int(i) for i in abstract.split(" ")]),
                    'vader_sentiment': torch.tensor(float(vader_sent)),
                    'bert_sentiment': torch.tensor(float(bert_sent))
                    }
        #
        # loading and preparing behaviors
        #
        # padding for news
        padding = {
            'category': torch.tensor(0),
            'subcategory': torch.tensor(0),
            'title': torch.tensor([0] * config.num_words_title),
            'abstract': torch.tensor([0] * config.num_words_abstract),
            'vader_sentiment': torch.tensor(0.0), 
            'bert_sentiment': torch.tensor(0.0)
        }

        with open(behavior_path, 'r') as file:
            behaviors = file.readlines()
            for behavior in tqdm(behaviors):
                uid, hist, candidates, clicks = behavior.split("\t")
                user = torch.tensor(int(uid))
                if hist:
                    history = [news_parsed[i] for i in hist.split(" ")]
                    if len(history) > config.max_history: 
                        history = history[:config.max_history]
                    else:
                        repeat = config.max_history - len(history)
                        history = [padding]*repeat + history
                else:
                    history = [padding]*config.max_history
                candidates = [news_parsed[i] for i in candidates.split(" ")]
                labels = torch.tensor([int(i) for i in clicks.split(" ")])
                self.behaviors_parsed.append(
                    {
                        'user': user,
                        'h_title': torch.stack([h['title'] for h in history]),
                        'h_abstract': torch.stack([h['abstract'] for h in history]),
                        'h_category': torch.stack([h['category'] for h in history]),
                        'h_subcategory': torch.stack([h['subcategory'] for h in history]),
                        'h_vader_sentiment': torch.stack([h['vader_sentiment'] for h in history]),
                        'h_bert_sentiment': torch.stack([h['bert_sentiment'] for h in history]),
                        'history_length': torch.tensor(len(history)),
                        'c_title': torch.stack([c['title'] for c in candidates]),
                        'c_abstract': torch.stack([c['abstract'] for c in candidates]),
                        'c_category': torch.stack([c['category'] for c in candidates]),
                        'c_subcategory': torch.stack([c['subcategory'] for c in candidates]),
                        'c_vader_sentiment': torch.stack([c['vader_sentiment'] for c in candidates]),
                        'c_bert_sentiment': torch.stack([c['bert_sentiment'] for c in candidates]),
                        'labels': labels
                    }
                )

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        return self.behaviors_parsed[idx]
