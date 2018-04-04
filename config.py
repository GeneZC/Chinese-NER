# -*- coding: utf-8 -*-
# author: Gene_ZC


class Config():
    train_path = "data/train.txt"
    val_path = "data/val.txt"
    test_path = "data/test.txt"
    score_file = "eval/temp/score.txt"
    mapping_path = 'models/mapping.pkl'
    tag_scheme = "iobes" # "iobes" or "iob"
    lower = True
    zeros = True # replace digits with 0s
    word_dim = 100  # character embedding dimension
    word_lstm_dim = 100  # character lstm hidden layer dimension
    char_dim = 50 # character embedding dimension
    char_lstm_dim = 50 # character lstm hidden layer dimension
    pre_emb = 'embedding/vectors.txt' # pretrained word embedding
    all_emb = True # load all embedding
    use_crf = True # use crf layer
    dropout = 0.5 # dropout rate
    lr = 0.002 # learning rate
    loss_file = "loss.txt"
    name = 'ner_lr'
    char_mode = 'CNN'
    reuse = False
