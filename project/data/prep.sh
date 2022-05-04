#!/bin/bash
mkdir MIND
mkdir MIND/train
mkdir MIND/test
mkdir word_embeddings
mkdir train
mkdir test

# download training data
wget "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip" -O temp.zip
unzip temp.zip -d MIND/train
rm temp.zip

# download test data
wget "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip" -O temp.zip
unzip temp.zip -d MIND/test
rm temp.zip

# download glove embeddings 
wget "http://nlp.stanford.edu/data/glove.840B.300d.zip" -O temp.zip
unzip temp.zip -d word_embeddings
rm temp.zip

# preprocess train-set impression logs
python parse_behavior.py --in-file MIND/train/behaviors.tsv --out-dir train --mode train

# preprocess test-set impression logs
python parse_behavior.py --in-file MIND/test/behaviors.tsv --out-dir test --mode test --user2int train/user2int.tsv 

# preprocess  train-set news-content
python parse_news.py --in-file MIND/train/news.tsv --out-dir train --mode train --word-embeddings word_embeddings/glove.840B.300d.txt

# preprocess test-set news-content
python parse_news.py --in-file MIND/test/news.tsv --out-dir test --mode test --word-embeddings word_embeddings/glove.840B.300d.txt --embedding-weights train/embedding_weights.csv  --word2int train/word2int.tsv --category2int train/category2int.tsv  