# -*- coding:utf-8 -*-
# author: Gene_ZC

def preprocess(filename):
    temp_str = ''
    flag = False
    f_save = open('prepro.txt', 'a+')
    with open(filename, 'r') as f:
        for line in f:
            if not line:
                continue
            else:
                line = line.strip().split('\t')[1]
                segment_list = line.split('  ')
                for item in segment_list:
                    word, feature = item.split('/')[0], item.split('/')[1]
                    if word.startswith('['):
                        temp_str += word[1:]
                        flag = True
                    elif feature[-3] == ']' and flag:
                        temp_str += word
                        feature = [-2:]
                        f_save.write(temp_str)
                        flag == False
                    elif 

if __name__ == '__main__':
    preprocess('raw.txt')