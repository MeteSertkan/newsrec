import argparse
from os import path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
import random
from nltk.tokenize import word_tokenize
import numpy as np
import csv
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='news file', required=True)
parser.add_argument('--out-dir', action='store', dest='out_dir',
                    help='parsed/pre-processed content dir', required=True)
parser.add_argument('--mode', action='store', dest='mode',
                    help='train or test', required=True)
parser.add_argument('--max-title', action='store', dest='max_title',
                    help='max title length', default=20)
parser.add_argument('--max-abstract', action='store', dest='max_abstract',
                    help='max abstract length', default=50)
parser.add_argument('--word-embeddings', action='store', dest='word_embeddings',
                    help='pre-trained word embeddings', required=True)
parser.add_argument('--word2int', action='store', dest='word2int',
                    help='word to idx map')
parser.add_argument('--embedding-weights', action='store', dest='embedding_weights',
                    help='word embedding weights')
parser.add_argument('--category2int', action='store', dest='category2int',
                    help='category to idx map')
args = parser.parse_args()


# generate word2int + extract embedding weights
def process_word_embeddings(word_embeddings_file):
    with open(word_embeddings_file, 'r') as wf:
        print("preparing/processing word-embeddings") 
        word_embeddings = wf.readlines()
        embeddings_map = {}
        for word_embedding in tqdm(word_embeddings):
            wdims = word_embedding.split(" ")
            embeddings_map[wdims[0]] = " ".join(wdims[1:])
        # np.save(path.join(out_dir, "embedding_weights.npy"), np.array(weights, dtype=np.float32))
        # with open(path.join(out_dir, 'word2int.tsv'), 'w') as file:  
        #    word_writer = csv.writer(file, delimiter='\t')
        #    for key, value in word2int.items():
        #        word_writer.writerow([key, value])
        return embeddings_map

def load_idx_map_as_dict(file_name):
    with open(file_name, 'r') as file:
        dictionary = {}
        lines = file.readlines()
        for line in tqdm(lines):
            key, value = line.strip().split("\t")
            dictionary[key] = value
        return dictionary

def load_embedding_weights(file_name):
    embedding_weights = []
    with open(file_name, 'r') as file: 
        lines = file.readlines()
        for line in tqdm(lines):
            embedding_weights.append(line)
        return embedding_weights

# prep embedings/vocab
embeddings = process_word_embeddings(args.word_embeddings)

    
# parse news 
with open(args.in_file, 'r') as in_file:
    with open(path.join(args.out_dir, 'parsed_news.tsv'), 'w') as news_file:  
        news_writer = csv.writer(news_file, delimiter='\t')
        print("preparing/processing news content")
        news_collection = in_file.readlines()
        news2int = {}
        # sentiment analyzer
        dsb_sentiment_classifier = pipeline('sentiment-analysis')
        vader_sentiment_classifier = SentimentIntensityAnalyzer()
        # max title/abstract length
        max_title_length = int(args.max_title)
        max_abstract_length = int(args.max_abstract)
        if args.mode == "train": 
            category2int = {}
            word2int = {}
            embedding_weights = []
        else:
            category2int = load_idx_map_as_dict(args.category2int)
            word2int = load_idx_map_as_dict(args.word2int)
            embedding_weights = load_embedding_weights(args.embedding_weights)
            
        # iterate over news
        for news in tqdm(news_collection):
            newsid, category, subcategory, title, abstract, _, _, _ = news.strip().split("\t")
            if newsid not in news2int:
                news2int[newsid] = len(news2int) + 1
            else:
                continue
            # category to int
            if category not in category2int:
                if(args.mode == "train"):
                    category2int[category] = len(category2int) + 1
                    category_id = category2int[category]
                else:
                    category_id = 0
            else: 
                category_id = category2int[category]
            if subcategory not in category2int:
                if(args.mode == "train"):
                    category2int[subcategory] = len(category2int) + 1
                    subcategory_id = category2int[subcategory]
                else:
                    subcategory_id = 0
            else: 
                subcategory_id = category2int[subcategory]
            # parse/prep title --> to token ids
            # crop at max-title or pad to max-title
            title_tokens = word_tokenize(title.strip().lower())
            title_word_idxs = []
            for token in title_tokens:
                if token not in embeddings:
                    continue
                if token not in word2int:
                    word2int[token] = str(len(word2int) + 1)
                    embedding_weights.append(embeddings[token])
                title_word_idxs.append(word2int[token])
            # title_word_idxs = [word2int[token] for token in title_tokens if token in word2int]
            if len(title_word_idxs) > max_title_length:
                title_word_idxs = title_word_idxs[:max_title_length]
            else:
                title_word_idxs = title_word_idxs + ["0"]*(max_title_length-len(title_word_idxs))
            title_word_idxs_str = " ".join(title_word_idxs)
            # parse/prep abstract --> to token ids
            # crop at max-abstract or pad to max-abstract
            abstract_tokens = word_tokenize(abstract.strip().lower())
            abstract_word_idxs = []
            for token in abstract_tokens:
                if token not in embeddings:
                    continue
                if token not in word2int:
                    word2int[token] = str(len(word2int) + 1)
                    embedding_weights.append(embeddings[token])
                abstract_word_idxs.append(word2int[token])
            if len(abstract_word_idxs) > max_abstract_length:
                abstract_word_idxs = abstract_word_idxs[:max_abstract_length]
            else:
                abstract_word_idxs = abstract_word_idxs + ["0"]*(max_abstract_length-len(abstract_word_idxs))
            abstract_word_idxs_str = " ".join(abstract_word_idxs)
            # calc sentiments scores
            # vader
            vs = vader_sentiment_classifier.polarity_scores(title.strip())
            vader_sentiment = vs['compound']
            # bert
            dsbs_label, dsbs_score = dsb_sentiment_classifier(title.strip())[0].values()
            if(dsbs_label == "POSITIVE"):
                bert_sentiment = (1-dsbs_score)*(-1) + dsbs_score
            else:
                bert_sentiment = (dsbs_score)*(-1) + (1-dsbs_score)
            # prepare output
            news_writer.writerow([
                newsid,
                category_id,
                subcategory_id,
                title_word_idxs_str,
                abstract_word_idxs_str,
                vader_sentiment,
                bert_sentiment
            ])
        if args.mode == "train":
            with open(path.join(args.out_dir, 'category2int.tsv'), 'w') as file:  
                cat_writer = csv.writer(file, delimiter='\t')
                for key, value in category2int.items():
                    cat_writer.writerow([key, value])
        with open(path.join(args.out_dir, 'word2int.tsv'), 'w') as file:
            word_writer = csv.writer(file, delimiter='\t')
            for key, value in word2int.items():
                word_writer.writerow([key, value])
        with open(path.join(args.out_dir, 'embedding_weights.csv'), 'w') as file:
            for weights in embedding_weights:
                file.write(weights)