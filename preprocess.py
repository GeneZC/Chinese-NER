# -*- coding:utf-8 -*-
# author: Gene_ZC

import codecs
import gensim
import random
import os
import pynlpir

def make_dev(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lst = f.read().split('\n\n')
        length = len(lst)
        random.shuffle(lst)
        lst = lst[:int(length/10)]
        
    with open('data/char_dev.txt', 'w', encoding='utf-8') as dev:
        for item in lst:
            dev.write(item+'\n\n') 

def create_corpus(filename):
    f_save = open('data/text8', 'w', encoding='utf-8')
    count = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                continue
            else:
                word = line.split(' ')[0]
                f_save.write(' '+word)
                count += 1
    f_save.write('\n')
    f_save.close()
    print('word count:' + str(count))

def preprocess(filename):
    f_save = open('data/char_test.txt', 'w', encoding='utf-8')
    pynlpir.open()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            lst = line.rstrip().split(' ')
            for item in lst:
                c, t = item.split('/')
                if t == 'o':
                    c = pynlpir.segment(c, pos_tagging=False)
                    for i, x in enumerate(c):
                        f_save.write(x+' '+'O'+'\n')
                elif t == 'ns':
                    c = pynlpir.segment(c, pos_tagging=False)
                    for i, x in enumerate(c):
                        if i == 0:
                            f_save.write(x+' '+'B-LOC'+'\n')
                        else:
                            f_save.write(x+' '+'I-LOC'+'\n')
                elif t == 'nt':
                    c = pynlpir.segment(c, pos_tagging=False)
                    for i, x in enumerate(c):
                        if i == 0:
                            f_save.write(x+' '+'B-ORG'+'\n')
                        else:
                            f_save.write(x+' '+'I-ORG'+'\n')
                elif t == 'nr':
                    c = pynlpir.segment(c, pos_tagging=False)
                    for i, x in enumerate(c):
                        if i == 0:
                            f_save.write(x+' '+'B-PER'+'\n')
                        else:
                            f_save.write(x+' '+'I-PER'+'\n')
            f_save.write('\n')
    f_save.close()

def test():
    model = gensim.models.KeyedVectors.load_word2vec_format('fastText/vectors.txt', binary=False)
    sim = model.most_similar('北京理工大学', topn=10)
    print(sim)

if __name__ == '__main__':
    # preprocess('data/testright1.txt')
    # create_corpus('data/char_train.txt')
    # make_dev('data/char_train.txt')
    test()
