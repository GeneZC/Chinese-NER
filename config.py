# -*- coding: utf-8 -*-
# author: Gene_ZC

class Config():
    def __init__(self):
        self.train_path = "dataset/eng.train"
        self.dev_path = "dataset/eng.testa"
        self.test_path = "dataset/eng.testb"
        self.test_train_path = "dataset/eng.train54019"
        self.score_file = "evaluation/temp/score.txt"
        self.tag_scheme = "iobes" # "iobes" or "iob"
        self.zeros = True # replace digits with 0s
        self.word_dim = 100  # character embedding dimension
        self.word_lstm_dim = 100  # character lstm hidden layer dimension
        self.word_bidirection = True  # character level lstm bi-directional
        self.char_dim = 25 # character embedding dimension
        self.char_lstm_dim = 25 # character lstm hidden layer dimension
        self.char_bidirection = True # character level lstm bi-directional
        self.pre_emb = 'embedding/vectors.txt' # pretrained word embedding
        self.all_emb = True # load all embedding
        self.use_crf = True # use crf layer
        self.dropout = 0.5 # dropout rate
        self.lr = 0.002 # learning rate
        self.loss_file = "loss.txt"
        self.name = 'ner'
        self.char_mode = 'CNN'
