# -*- coding: utf8 -*- 
from __future__ import print_function
import itertools
from collections import OrderedDict
import loader
import torch
import time
import pickle
import codecs
from torch.autograd import Variable
import sys
from utils import *
from loader import *
from model import BiLSTM_CRF
from config import Config

config = Config()

parameters = OrderedDict()
parameters['tag_scheme'] = config.tag_scheme
parameters['lower'] = config.lower
parameters['zeros'] = config.zeros
parameters['char_dim'] = config.char_dim
parameters['char_lstm_dim'] = config.char_lstm_dim
parameters['word_dim'] = config.word_dim
parameters['word_lstm_dim'] = config.word_lstm_dim
parameters['pre_emb'] = config.pre_emb
parameters['all_emb'] = config.all_emb
parameters['crf'] = config.use_crf
parameters['dropout'] = config.dropout
parameters['reload'] = config.reuse
parameters['name'] = config.name
parameters['char_mode'] = config.char_mode
parameters['use_gpu'] = torch.cuda.is_available()

use_gpu = parameters['use_gpu']
name = parameters['name']
model_name = os.path.join(models_path, name) #get_name(parameters)
tmp_model = model_name + '.tmp'


assert os.path.isfile(config.train_path)
assert os.path.isfile(config.val_path)
assert os.path.isfile(config.test_path)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

train_sentences = loader.load_sentences(config.train_path, lower, zeros)
val_sentences = loader.load_sentences(config.val_path, lower, zeros)
test_sentences = loader.load_sentences(config.test_path, lower, zeros)

update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(val_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

dico_words_train = word_mapping(train_sentences, lower)[0]

dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in test_sentences])
        ) if not parameters['all_emb'] else None
    )

dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
val_data = prepare_dataset(
    val_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

print("%i / %i / %i sentences in train / val / test." % (
    len(train_data), len(val_data), len(test_data)))

all_word_embeds = {}
for i, line in enumerate(codecs.open(config.pre_emb, 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == parameters['word_dim'] + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), config.word_dim))

for w in word_to_id:
    if w in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

with open(mapping_file, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'parameters': parameters,
        'word_embeds': word_embeds
    }
    pickle.dump(mappings, f)

print('word_to_id length: ', len(word_to_id))
model = BiLSTM_CRF(vocab_size=len(word_to_id),
                   tag_to_ix=tag_to_id,
                   embedding_dim=parameters['word_dim'],
                   hidden_dim=parameters['word_lstm_dim'],
                   use_gpu=use_gpu,
                   char_to_ix=char_to_id,
                   pre_word_embeds=word_embeds,
                   use_crf=parameters['crf'],
                   char_mode=parameters['char_mode'])

if parameters['reload']:
    model.load_state_dict(torch.load(model_name))
if use_gpu:
    model.cuda()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
losses = []
loss = 0.0
best_val_F = -1.0
best_test_F = -1.0
all_F = [[0, 0]]
plot_every = 100
eval_every = 2000
sys.stdout.flush()

def evaluating(model, datas, best_F):
    prediction = []
    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
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

        dwords = Variable(torch.LongTensor(data['words']))
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, chars2_length, d)
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
    predf = eval_temp + '/pred.' + name
    scoref = eval_temp + '/score.' + name

    with codecs.open(predf, 'w', encoding='utf8') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    eval_lines = [l.rstrip() for l in codecs.open(scoref, 'r', 'utf8')]

    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_F = float(line.strip().split()[-1])
            if new_F > best_F:
                best_F = new_F
                save = True
                print('the best F is ', new_F)

    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    ))
    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        ))
    return best_F, new_F, save

model.train(True)
for epoch in range(1, 1001):
    count = 0
    for i, index in enumerate(np.random.permutation(len(train_data))):
        tr = time.time()
        count += 1
        data = train_data[index]
        model.zero_grad()

        sentence_in = data['words']
        sentence_in = Variable(torch.LongTensor(sentence_in))
        tags = data['tags']
        chars2 = data['chars']

        ######### char lstm
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

        # ######## char cnn
        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))


        targets = torch.LongTensor(tags)
        if use_gpu:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, chars2_length, d)
        loss += neg_log_likelihood.data[0] / len(data['words'])
        neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()

        if count % plot_every == 0:
            loss /= plot_every
            print('Epoch: ' + str(epoch) + '-----' + 'Iteration: ' + str(count) + '-------' + 'Loss: ' + str(loss))
            loss = 0.

        if count % (eval_every) == 0:
            model.train(False)
            print('train dataset evaluation -----')
            best_val_F, new_val_F, save = evaluating(model, val_data, best_val_F)
            if save:
                torch.save(model.state_dict(), model_name)
            print('test dataset evaluation -----')
            best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F)
            sys.stdout.flush()

            all_F.append([new_val_F, new_test_F])
            model.train(True)

        if count % len(train_data) == 0:
            adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))

