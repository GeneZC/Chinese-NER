# -*- coding: utf8 -*-
from __future__ import print_function
import torch
import pickle
from torch.autograd import Variable
from config import Config
from utils import *
from loader import *
from model import BiLSTM_CRF
import jieba
jieba.initialize()

model_name = os.path.join(models_path, Config.name) #get_name(parameters)

mapping_file = Config.mapping_path

with open(mapping_file, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    mappings = u.load()

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
id_to_tag = {k[1]: k[0] for k in tag_to_id.items()}
char_to_id = mappings['char_to_id']
parameters = mappings['parameters']
word_embeds = mappings['word_embeds']

use_gpu = torch.cuda.is_available()

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

model = BiLSTM_CRF(vocab_size=len(word_to_id),
                   tag_to_ix=tag_to_id,
                   embedding_dim=parameters['word_dim'],
                   hidden_dim=parameters['word_lstm_dim'],
                   use_gpu=use_gpu,
                   char_to_ix=char_to_id,
                   pre_word_embeds=word_embeds,
                   use_crf=parameters['crf'],
                   char_mode=parameters['char_mode'])

model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))

if use_gpu:
    model.cuda()
model.eval()

def evaluate_line(model, sentence):
    prediction = []
    seg = jieba.lcut(sentence)
    data = prepare_sentence(seg, word_to_id, char_to_id)
    str_words = data['str_words']
    words = data['words']
    chars2 = data['chars']
    if parameters['char_mode'] == 'LSTM':
        chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
        d = {}
        for i, ci in enumerate(chars2):
            for j, cj in enumerate(chars2_sorted):
                if ci == cj and not j in d and not i in d.values():
                    d[j] = i
                    continue
        chars2_length = [len(c) for c in chars2_sorted]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
        for i, c in enumerate(chars2_sorted):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))

    if parameters['char_mode'] == 'CNN':
        d = {}
        chars2_length = [len(c) for c in chars2]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
        for i, c in enumerate(chars2):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))

    dwords = Variable(torch.LongTensor(words))
    if use_gpu:
        val, out = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
    else:
        val, out = model(dwords, chars2_mask, chars2_length, d)
    predicted_id = out
    for (word, pred_id) in zip(str_words, predicted_id):
        line = ' '.join([word, id_to_tag[pred_id]])
        prediction.append(line)
    for item in prediction:
        print(item)

line = input('Please input a sentence.\n')
evaluate_line(model, line)


