# -*- coding:utf-8 -*-
# author: Gene_ZC

import codecs
import gensim
import random
import os
import jieba
jieba.initialize()

def make_dev(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lst = f.read().split('\n\n')
        length = len(lst)
        random.shuffle(lst)
        lst = lst[:int(length/10)]
        
    with open('data/dev.txt', 'w', encoding='utf-8') as dev:
        for item in lst:
            dev.write(item+'\n\n') 

def create_corpus(filename):
    f_save = open('data/text8', 'w', encoding='utf-8')
    count = 0
    temp_str = ''
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                lst = jieba.lcut(temp_str)
                for item in lst:
                    f_save.write(' ' + item)
                temp_str = ''
            else:
                word = line.split(' ')[0]
                temp_str += word
                count += 1
    f_save.write('\n')
    f_save.close()
    print('word count:' + str(count))

def test():
    model = gensim.models.KeyedVectors.load_word2vec_format('embedding/fasttext_vec.txt', binary=False)
    sim = model.most_similar('haha', topn=10)
    print(sim)

if __name__ == '__main__':
    # preprocess()
    # create_corpus('data/train.txt')
    # make_dev('data/train.txt')
    # test()
