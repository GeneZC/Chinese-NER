# -*- coding:utf-8 -*-
# author: Gene_ZC

import codecs
import gensim
import random
import os
import jieba
jieba.initialize()

def split_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lst = f.read().split('\n\n')
        length = len(lst)
        random.shuffle(lst)
        train_lst = lst[:int(length*4/5)+int(length/10)]
        val_lst = lst[int(length*4/5):int(length*4/5)+int(length/10)]
        test_lst = lst[int(length*4/5)+int(length/10):]
    with open('data/train.txt', 'w', encoding='utf-8') as train:
        for item in train_lst:
            train.write(item+'\n\n')

    with open('data/val.txt', 'w', encoding='utf-8') as val:
        for item in val_lst:
            val.write(item+'\n\n')

    with open('data/test.txt', 'w', encoding='utf-8') as test:
        for item in test_lst:
            test.write(item+'\n\n')       

def preprocess():
    temp_str = None
    flag = False
    f_save = open('data/prepro.txt', 'w', encoding='utf-8')
    filenames = os.listdir('./raw_corpus')
    for filename in filenames:
        with codecs.open(os.path.join('./raw_corpus',filename), 'r', encoding='gb18030', errors='ignore') as f:
            for line in f:
                if line == '\r\n':
                    continue
                line = line.split('\t')[1]
                segment_list = line.split('  ')[:-1]
                for item in segment_list:
                    word, feature = item.split('/')[0], item.split('/')[1]
                    if word.startswith('['):
                        temp_str = []
                        temp_str.append(word[1:])
                        flag = True
                    elif flag:
                        try:
                            anno = feature[-3]
                        except:
                            anno = ''
                        if anno == ']':
                            temp_str.append(word)
                            feature = feature[-2:]
                            for i, c in enumerate(''.join(temp_str)):
                                if i == 0:
                                    if feature == 'nt':  # ORG
                                        f_save.write(c + ' ' + 'B-' + 'ORG' + '\n')
                                    elif feature == 'ns':  # LOC
                                        f_save.write(c + ' ' + 'B-' + 'LOC' + '\n')
                                    elif feature == 'nr':  # PER
                                        f_save.write(c + ' ' + 'B-' + 'PER' + '\n')
                                else:
                                    if feature == 'nt':  # ORG
                                        f_save.write(c + ' ' + 'I-' + 'ORG' + '\n')
                                    elif feature == 'ns':  # LOC
                                        f_save.write(c + ' ' + 'I-' + 'LOC' + '\n')
                                    elif feature == 'nr':  # PER
                                        f_save.write(c + ' ' + 'I-' + 'PER' + '\n')
                            flag = False
                            temp_str = []
                        else:
                            temp_str.append(word)

                    else:
                        if feature == 'nt':  # ORG
                            for i, c in enumerate(word):
                                if i == 0:
                                    f_save.write(c + ' ' + 'B-' + 'ORG' + '\n')
                                else:
                                    f_save.write(c + ' ' + 'I-' + 'ORG' + '\n')
                        elif feature == 'ns':  # LOC
                            for i, c in enumerate(word):
                                if i == 0:
                                    f_save.write(c + ' ' + 'B-' + 'LOC' + '\n')
                                else:
                                    f_save.write(c + ' ' + 'I-' + 'LOC' + '\n')
                        elif feature == 'nr':  # PER
                            for i, c in enumerate(word):
                                if i == 0:
                                    f_save.write(c + ' ' + 'B-' + 'PER' + '\n')
                                else:
                                    f_save.write(c + ' ' + 'I-' + 'PER' + '\n')
                        else:
                            f_save.write(word + ' ' + 'O' + '\n')
                f_save.write('\n')
    f_save.close()

def preprocess_char():
    temp_str = None
    flag = False
    f_save = open('data/prepro_char.txt', 'w', encoding='utf-8')
    filenames = os.listdir('./raw_corpus')
    for filename in filenames:
        with codecs.open(os.path.join('./raw_corpus',filename), 'r', encoding='gb18030', errors='ignore') as f:
            for line in f:
                if line == '\r\n':
                    continue
                line = line.split('\t')[1]
                segment_list = line.split('  ')[:-1]
                for item in segment_list:
                    word, feature = item.split('/')[0], item.split('/')[1]
                    if word.startswith('['):
                        temp_str = []
                        temp_str.append(word[1:])
                        flag = True
                    elif flag:
                        try:
                            anno = feature[-3]
                        except:
                            anno = ''
                        if anno == ']':
                            temp_str.append(word)
                            feature = feature[-2:]
                            # f_save.write(temp_str)
                            for i, c in enumerate(''.join(temp_str)):
                                if i == 0:
                                    if feature == 'nt':  # ORG
                                        f_save.write(c + ' ' + 'B-' + 'ORG' + '\n')
                                    elif feature == 'ns':  # LOC
                                        f_save.write(c + ' ' + 'B-' + 'LOC' + '\n')
                                    elif feature == 'nr':  # PER
                                        f_save.write(c + ' ' + 'B-' + 'PER' + '\n')
                                else:
                                    if feature == 'nt':  # ORG
                                        f_save.write(c + ' ' + 'I-' + 'ORG' + '\n')
                                    elif feature == 'ns':  # LOC
                                        f_save.write(c + ' ' + 'I-' + 'LOC' + '\n')
                                    elif feature == 'nr':  # PER
                                        f_save.write(c + ' ' + 'I-' + 'PER' + '\n')
                            flag = False
                            temp_str = []
                        else:
                            temp_str.append(word)

                    else:

                        if feature == 'nt':  # ORG
                            for i, c in enumerate(word):
                                if i == 0:
                                    f_save.write(c + ' ' + 'B-' + 'ORG' + '\n')
                                else:
                                    f_save.write(c + ' ' + 'I-' + 'ORG' + '\n')
                        elif feature == 'ns':  # LOC
                            for i, c in enumerate(word):
                                if i == 0:
                                    f_save.write(c + ' ' + 'B-' + 'LOC' + '\n')
                                else:
                                    f_save.write(c + ' ' + 'I-' + 'LOC' + '\n')
                        elif feature == 'nr':  # PER
                            for i, c in enumerate(word):
                                if i == 0:
                                    f_save.write(c + ' ' + 'B-' + 'PER' + '\n')
                                else:
                                    f_save.write(c + ' ' + 'I-' + 'PER' + '\n')
                        else:
                            for i, c in enumerate(word):
                                f_save.write(c + ' ' + 'O' + '\n')
                f_save.write('\n')
    f_save.close()

def create_corpus(filename):
    f_save = open('data/text8', 'w', encoding='utf-8')
    count = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                continue
            else:
                word = line.split(' ')[0]
                count += 1
                f_save.write(' ' + word)
    f_save.write('\n')
    f_save.close()
    print('word count:' + str(count))

def test():
    model = gensim.models.KeyedVectors.load_word2vec_format('embedding/fasttext_vec.txt', binary=False)
    sim = model.most_similar('泽民', topn=10)
    print(sim)

if __name__ == '__main__':
    # preprocess()
    # preprocess_char()
    # create_corpus('data/prepro.txt')
    split_dataset('data/prepro.txt')
    # test()
