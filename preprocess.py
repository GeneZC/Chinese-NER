# -*- coding:utf-8 -*-
# author: Gene_ZC

import codecs
import gensim

def preprocess(filename):
    temp_str = None
    flag = False
    f_save = open('data/prepro.txt', 'a+', encoding='utf-8')
    with codecs.open(filename, 'r', encoding='gbk') as f:
        for line in f:
            if line == '\r\n':
                continue
            line = line.split('\t')[1]
            segment_list = line.split('  ')[1:-1]
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
                        for i, w in enumerate(temp_str):
                            if i == 0:
                                if feature == 'nt':  # ORG
                                    f_save.write(w + ' ' + 'B-' + 'ORG' + '\n')
                                elif feature == 'ns':  # LOC
                                    f_save.write(w + ' ' + 'B-' + 'LOC' + '\n')
                                elif feature == 'nr':  # PER
                                    f_save.write(w + ' ' + 'B-' + 'PER' + '\n')
                            else:
                                if feature == 'nt':  # ORG
                                    f_save.write(w + ' ' + 'I-' + 'ORG' + '\n')
                                elif feature == 'ns':  # LOC
                                    f_save.write(w + ' ' + 'I-' + 'LOC' + '\n')
                                elif feature == 'nr':  # PER
                                    f_save.write(w + ' ' + 'I-' + 'PER' + '\n')
                        flag = False
                        temp_str = []
                    else:
                        temp_str.append(word)

                else:
                    if feature == 'nt':  # ORG
                        f_save.write(word + ' ' + 'B-' + 'ORG' + '\n')
                    elif feature == 'ns':  # LOC
                        f_save.write(word + ' ' + 'B-' + 'LOC' + '\n')
                    elif feature == 'nr':  # PER
                        f_save.write(word + ' ' + 'B-' + 'PER' + '\n')
                    else:
                        f_save.write(word + ' ' + 'O' + '\n')
            f_save.write('\n')
    f_save.close()

def create_corpus(filename):
    f_save = open('Glove/text8', 'a+', encoding='utf-8')
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
    model = gensim.models.KeyedVectors.load_word2vec_format('vectors.txt', binary=False)
    sim = model.most_similar('泽民', topn=10)
    print(sim)

if __name__ == '__main__':
    # preprocess('data/raw.txt')
    # create_corpus('data/prepro.txt')
    test()
