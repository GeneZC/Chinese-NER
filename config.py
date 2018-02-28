# -*- coding: utf-8 -*-
# author: Gene_ZC

class Config():
    train_path = "data/train.txt"
    dev_path = "data/dev.txt"
    test_path = "data/test.txt"
    test_train_path = "data/dev.txt"
    score_file = "eval/temp/score.txt"
    tag_scheme = "iobes" # "iobes" or "iob"
    zeros = True # replace digits with 0s
    word_dim = 100  # character embedding dimension
    word_lstm_dim = 100  # character lstm hidden layer dimension
    word_bidirection = True  # character level lstm bi-directional
    char_dim = 25 # character embedding dimension
    char_lstm_dim = 25 # character lstm hidden layer dimension
    char_bidirection = True # character level lstm bi-directional
    pre_emb = 'embedding/vectors.txt' # pretrained word embedding
    all_emb = True # load all embedding
    use_crf = True # use crf layer
    dropout = 0.5 # dropout rate
    lr = 0.002 # learning rate
    loss_file = "loss.txt"
    name = 'ner'
    char_mode = 'CNN'
