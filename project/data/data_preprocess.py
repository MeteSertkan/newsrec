import argparse
import yaml
from dotmap import DotMap
import pandas as pd
import swifter
import json
import math
from tqdm import tqdm
import pathlib
from os import path
from pathlib import Path
import random
from nltk.tokenize import word_tokenize
import numpy as np
import csv
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

dsb_sentiment_classifier = pipeline('sentiment-analysis')
vader_sentiment_classifier = SentimentIntensityAnalyzer()

def parse_behaviors(config, source, target, user2int_path, mode):
    """
    Parse behaviors file.
    Args:
        source: source behaviors file
        target: target behaviors file
        user2int_path: path for saving user2int file
        mode: train / test
    """
    print(f"Parse {source}")

    behaviors = pd.read_table(
        source,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors.clicked_news.fillna(' ', inplace=True)
    behaviors.impressions = behaviors.impressions.str.split()

    if mode == 'train':
        user2int = {}
        for row in behaviors.itertuples(index=False):
            if row.user not in user2int:
                user2int[row.user] = len(user2int) + 1
        
        pd.DataFrame(user2int.items(), columns=['user',
                                                'int']).to_csv(user2int_path,
                                                            sep='\t',
                                                            index=False)
        print(
            f'Please modify `num_users` in `src/config.py` into 1 + {len(user2int)}'  # noqa: E501
        )

        for row in behaviors.itertuples():
            behaviors.at[row.Index, 'user'] = user2int[row.user]

        for row in tqdm(behaviors.itertuples(), desc="Balancing data"):
            positive = iter([x for x in row.impressions if x.endswith('1')])
            negative = [x for x in row.impressions if x.endswith('0')]
            random.shuffle(negative)
            negative = iter(negative)
            pairs = []
            try:
                while True:
                    pair = [next(positive)]
                    for _ in range(config.negative_sampling_ratio):
                        pair.append(next(negative))
                    pairs.append(pair)
            except StopIteration:
                pass
            behaviors.at[row.Index, 'impressions'] = pairs

        behaviors = behaviors.explode('impressions').dropna(
            subset=["impressions"]).reset_index(drop=True)

    if mode == 'test':
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                behaviors.at[row.Index, 'user'] = 0
        print(f'User miss rate: {user_missed/user_total:.4f}')

    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (' '.join([e.split('-')[0] for e in x]), ' '.join(
                [e.split('-')[1] for e in x]))).tolist())
    behaviors.to_csv(
        target,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])


def parse_news(config, source, target, category2int_path,
               word2int_path, entity2int_path, mode):
    """
    Parse news for training set and test set
    Args:
        source: source news file
        target: target news file
        if mode == 'train':
            category2int_path, word2int_path, entity2int_path: Path to save
        elif mode == 'test':
            category2int_path, word2int_path, entity2int_path: Path to load from
    """
    print(f"Parse {source}")
    news = pd.read_table(source,
                         header=None,
                         usecols=[0, 1, 2, 3, 4, 6, 7],
                         quoting=csv.QUOTE_NONE,
                         names=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract', 'title_entities', 'abstract_entities'
                         ])  # TODO try to avoid csv.QUOTE_NONE
    news.title_entities.fillna('[]', inplace=True)
    news.abstract_entities.fillna('[]', inplace=True)
    news.fillna(' ', inplace=True)

    def parse_row(row):
        new_row = [
            row.id,
            category2int[row.category] if row.category in category2int else 0,
            category2int[row.subcategory]
            if row.subcategory in category2int else 0,
            [0] * config.num_words_title, [0] * config.num_words_abstract,
            [0] * config.num_words_title, [0] * config.num_words_abstract,
            0, 0
        ]

        # Calculate local entity map (map lower single word to entity)
        local_entity_map = {}
        for e in json.loads(row.title_entities):
            if e['Confidence'] > config.entity_confidence_threshold and e[
                    'WikidataId'] in entity2int:
                for x in ' '.join(e['SurfaceForms']).lower().split():
                    local_entity_map[x] = entity2int[e['WikidataId']]
        for e in json.loads(row.abstract_entities):
            if e['Confidence'] > config.entity_confidence_threshold and e[
                    'WikidataId'] in entity2int:
                for x in ' '.join(e['SurfaceForms']).lower().split():
                    local_entity_map[x] = entity2int[e['WikidataId']]

        try:
            for i, w in enumerate(word_tokenize(row.title.lower())):
                if w in word2int:
                    new_row[3][i] = word2int[w]
                    if w in local_entity_map:
                        new_row[5][i] = local_entity_map[w]
        except IndexError:
            pass

        try:
            for i, w in enumerate(word_tokenize(row.abstract.lower())):
                if w in word2int:
                    new_row[4][i] = word2int[w]
                    if w in local_entity_map:
                        new_row[6][i] = local_entity_map[w]
        except IndexError:
            pass

        # ASSIGN SENTIMENT SCORE [-1,1]
        # vaderSentiment
        vs = vader_sentiment_classifier.polarity_scores(row.title)
        new_row[7] = vs['compound']

        # distilbert fine tuned on SST2
        dsbs_label, dsbs_score = dsb_sentiment_classifier(row.title)[0].values()
        if(dsbs_label == "POSITIVE"):
            new_row[8] = (1-dsbs_score)*(-1) + dsbs_score
        else:
            new_row[8] = (dsbs_score)*(-1) + (1-dsbs_score)

        return pd.Series(new_row,
                         index=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract', 'title_entities', 'abstract_entities',
                             'vader_sentiment', 'distillbert_sst2_sentiment'
                         ])

    if mode == 'train':
        category2int = {}
        word2int = {}
        word2freq = {}
        entity2int = {}
        entity2freq = {}

        for row in news.itertuples(index=False):
            if row.category not in category2int:
                category2int[row.category] = len(category2int) + 1
            if row.subcategory not in category2int:
                category2int[row.subcategory] = len(category2int) + 1

            for w in word_tokenize(row.title.lower()):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1
            for w in word_tokenize(row.abstract.lower()):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1

            for e in json.loads(row.title_entities):
                times = len(e['OccurrenceOffsets']) * e['Confidence']
                if times > 0:
                    if e['WikidataId'] not in entity2freq:
                        entity2freq[e['WikidataId']] = times
                    else:
                        entity2freq[e['WikidataId']] += times

            for e in json.loads(row.abstract_entities):
                times = len(e['OccurrenceOffsets']) * e['Confidence']
                if times > 0:
                    if e['WikidataId'] not in entity2freq:
                        entity2freq[e['WikidataId']] = times
                    else:
                        entity2freq[e['WikidataId']] += times

        for k, v in word2freq.items():
            if v >= config.word_freq_threshold:
                word2int[k] = len(word2int) + 1

        for k, v in entity2freq.items():
            if v >= config.entity_freq_threshold:
                entity2int[k] = len(entity2int) + 1

        parsed_news = news.swifter.apply(parse_row, axis=1)
        # TODO CONCAT HERE THE sentiment score parsed_news = pd.concat([parsed_news, sentiment_df], axis=1)
        parsed_news.to_csv(target, sep='\t', index=False)

        pd.DataFrame(category2int.items(),
                     columns=['category', 'int']).to_csv(category2int_path,
                                                         sep='\t',
                                                         index=False)
        print(
            f'Please modify `num_categories` in `src/config.py` into 1 + {len(category2int)}'  # noqa: E501
        )

        pd.DataFrame(word2int.items(), columns=['word',
                                                'int']).to_csv(word2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_words` in `src/config.py` into 1 + {len(word2int)}'  # noqa: E501
        )

        pd.DataFrame(entity2int.items(),
                     columns=['entity', 'int']).to_csv(entity2int_path,
                                                       sep='\t',
                                                       index=False)
        print(
            f'Please modify `num_entities` in `src/config.py` into 1 + {len(entity2int)}'  # noqa: E501
        )

    elif mode == 'test':
        category2int = dict(pd.read_table(category2int_path).values.tolist())
        # na_filter=False is needed since nan is also a valid word
        word2int = dict(
            pd.read_table(word2int_path, na_filter=False).values.tolist())
        entity2int = dict(pd.read_table(entity2int_path).values.tolist())

        parsed_news = news.swifter.apply(parse_row, axis=1)
        # TODO CONCAT HERE THE sentiment score parsed_news = pd.concat([parsed_news, sentiment_df], axis=1)
        parsed_news.to_csv(target, sep='\t', index=False)

    else:
        print('Wrong mode!')


def generate_word_embedding(config, source, target, word2int_path):
    """
    Generate from pretrained word embedding file
    If a word not in embedding file, initial its embedding by N(0, 1)
    Args:
        source: path of pretrained word embedding file, e.g. glove.840B.300d.txt # noqa: E501
        target: path for saving word embedding. Will be saved in numpy format
        word2int_path: vocabulary file when words in it will be searched in pretrained embedding file # noqa: E501
    """
    # na_filter=False is needed since nan is also a valid word
    # word, int
    word2int = pd.read_table(word2int_path, na_filter=False, index_col='word')
    source_embedding = pd.read_table(source,
                                     index_col=0,
                                     sep=' ',
                                     header=None,
                                     quoting=csv.QUOTE_NONE,
                                     names=range(config.word_embedding_dim))
    # word, vector
    source_embedding.index.rename('word', inplace=True)
    # word, int, vector
    merged = word2int.merge(source_embedding,
                            how='inner',
                            left_index=True,
                            right_index=True)
    merged.set_index('int', inplace=True)

    missed_index = np.setdiff1d(np.arange(len(word2int) + 1),
                                merged.index.values)
    missed_embedding = pd.DataFrame(data=np.random.normal(
        size=(len(missed_index), config.word_embedding_dim)))
    missed_embedding['int'] = missed_index
    missed_embedding.set_index('int', inplace=True)

    final_embedding = pd.concat([merged, missed_embedding]).sort_index()
    np.save(target, final_embedding.values)

    print(
        f'Rate of word missed in pretrained embedding: {(len(missed_index)-1)/len(word2int):.4f}'  # noqa: E501
    )


def transform_entity_embedding(config, source, target, entity2int_path):
    """
    Args:
        source: path of embedding file
        target: path of transformed embedding file in numpy format
        entity2int_path
    """
    entity_embedding = pd.read_table(source, header=None)
    entity_embedding['vector'] = entity_embedding.iloc[:,
                                                       1:101].values.tolist()
    entity_embedding = entity_embedding[[0, 'vector'
                                         ]].rename(columns={0: "entity"})

    entity2int = pd.read_table(entity2int_path)
    merged_df = pd.merge(entity_embedding, entity2int,
                         on='entity').sort_values('int')
    entity_embedding_transformed = np.random.normal(
        size=(len(entity2int) + 1, config.entity_embedding_dim))
    for row in merged_df.itertuples(index=False):
        entity_embedding_transformed[row.int] = row.vector
    np.save(target, entity_embedding_transformed)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        action='store',
        dest='config',
        help='data_config.yaml',
        required=True)
    
    parser.add_argument(
        '--mode',
        action='store',
        dest='mode',
        help='train, test, or all data',
        required=True)

    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)
        config = DotMap(config)

        if args.mode == 'train' or args.mode == 'all':
            print('Process data for training')

            pathlib.Path(config.train_target_dir).mkdir(parents=True, exist_ok=True)

            print('Parse behaviors')
            parse_behaviors(
                config,
                path.join(config.train_source_dir, 'behaviors.tsv'),
                path.join(config.train_target_dir, 'behaviors_parsed.tsv'),
                path.join(config.train_target_dir, 'user2int.tsv'),
                mode='train')

            print('Parse news')
            parse_news(
                config,
                path.join(config.train_source_dir, 'news.tsv'),
                path.join(config.train_target_dir, 'news_parsed.tsv'),
                path.join(config.train_target_dir, 'category2int.tsv'),
                path.join(config.train_target_dir, 'word2int.tsv'),
                path.join(config.train_target_dir, 'entity2int.tsv'),
                mode='train')

            print('Generate word embedding')
            generate_word_embedding(
                config,
                config.word_embedding_source,
                path.join(config.train_target_dir, 'pretrained_word_embedding.npy'),  # noqa: E501
                path.join(config.train_target_dir, 'word2int.tsv'))

            print('Transform entity embeddings')
            transform_entity_embedding(
                config,
                path.join(config.train_source_dir, 'entity_embedding.vec'),
                path.join(config.train_target_dir, 'pretrained_entity_embedding.npy'),  # noqa: E501
                path.join(config.train_target_dir, 'entity2int.tsv'))

        if args.mode == 'test' or args.mode == 'all':
            print('\nProcess data for test')

            pathlib.Path(config.test_target_dir).mkdir(parents=True, exist_ok=True)

            print('Parse behaviors')
            parse_behaviors(
                config,
                path.join(config.test_source_dir, 'behaviors.tsv'),
                path.join(config.test_target_dir, 'behaviors_parsed.tsv'),
                path.join(config.train_target_dir, 'user2int.tsv'),
                mode='test')

            print('Parse news')
            parse_news(
                config,
                path.join(config.test_source_dir, 'news.tsv'),
                path.join(config.test_target_dir, 'news_parsed.tsv'),
                path.join(config.train_target_dir, 'category2int.tsv'),
                path.join(config.train_target_dir, 'word2int.tsv'),
                path.join(config.train_target_dir, 'entity2int.tsv'),
                mode='test')

if __name__ == '__main__':
    cli_main()
