# -*- coding:utf-8 -*-
# author: GeneZC

import itertools
import loader
import torch
import time
import pickle
import matplotlib.pyplot as plt
import sys
import visdom
from torch.autograd import Variable
from collections import OrderedDict
from model import BiLSTM_CRF
from config import Config
from utils import *
from loader import *

class Trainpipeline():
    def __init__(self, opts):
        self.parameters = OrderedDict()
        self.train_path = opts.train_path
        self.dev_path = opts.dev_path
        self.test_path = opts.test_path
        self.test_train_path = opts.test_train_path
        self.parameters['tag_scheme'] = opts.tag_scheme
        self.parameters['zeros'] = opts.zeros
        self.parameters['char_dim'] = opts.char_dim
        self.parameters['char_lstm_dim'] = opts.char_lstm_dim
        self.parameters['char_bidirect'] = opts.char_bidirection
        self.parameters['word_dim'] = opts.word_dim
        self.parameters['word_lstm_dim'] = opts.word_lstm_dim
        self.parameters['word_bidirect'] = opts.word_bidirection
        self.parameters['pre_emb'] = opts.pre_emb
        self.parameters['all_emb'] = opts.all_emb
        self.parameters['crf'] = opts.use_crf
        self.parameters['dropout'] = opts.dropout
        self.parameters['name'] = opts.name
        self.parameters['char_mode'] = opts.char_mode
        self.learning_rate = opts.lr
        self.use_gpu = torch.cuda.is_available()
        self.mapping_file = mapping_path
        self.model_name = models_path + self.parameters['name']  # get_name(self.parameters)
        self.tmp_model = self.model_name + '.tmp'

        if not os.path.isfile(eval_script):
            raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
        if not os.path.exists(eval_temp):
            os.makedirs(eval_temp)
        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def load(self):
        zeros = self.parameters['zeros']
        tag_scheme = self.parameters['tag_scheme']

        train_sentences = loader.load_sentences(self.train_path, zeros)
        dev_sentences = loader.load_sentences(self.dev_path, zeros)
        test_sentences = loader.load_sentences(self.test_path, zeros)
        test_train_sentences = loader.load_sentences(self.test_train_path, zeros)

        update_tag_scheme(train_sentences, tag_scheme)
        update_tag_scheme(dev_sentences, tag_scheme)
        update_tag_scheme(test_sentences, tag_scheme)
        update_tag_scheme(test_train_sentences, tag_scheme)

        dico_words_train = word_mapping(train_sentences)[0]

        dico_words, word_to_id, id_to_word = augment_with_pretrained(
            dico_words_train.copy(),
            self.parameters['pre_emb'],
            list(itertools.chain.from_iterable(
                [[w[0] for w in s] for s in dev_sentences + test_sentences])
            ) if not self.parameters['all_emb'] else None
        )

        dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

        train_data = prepare_dataset(
            train_sentences, word_to_id, char_to_id, tag_to_id
        )
        dev_data = prepare_dataset(
            dev_sentences, word_to_id, char_to_id, tag_to_id
        )
        test_data = prepare_dataset(
            test_sentences, word_to_id, char_to_id, tag_to_id
        )
        test_train_data = prepare_dataset(
            test_train_sentences, word_to_id, char_to_id, tag_to_id
        )

        print("%i / %i / %i sentences in train / dev / test." % (
            len(train_data), len(dev_data), len(test_data)))

        all_word_embeds = {}
        for i, line in enumerate(codecs.open(self.parameters['pre_emb'], 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == self.parameters['word_dim'] + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), self.parameters['word_dim']))

        for w in word_to_id:
            if w in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w]

        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

        with open(self.mapping_file, 'wb') as f:
            mappings = {
                'word_to_id': word_to_id,
                'tag_to_id': tag_to_id,
                'char_to_id': char_to_id,
                'self.parameters': self.parameters,
                'word_embeds': word_embeds
            }
            pickle.dump(mappings, f)

        print('word_to_id: ', len(word_to_id))
        self.model = BiLSTM_CRF(vocab_size=len(word_to_id),
                           tag_to_ix=tag_to_id,
                           embedding_dim=self.parameters['word_dim'],
                           hidden_dim=self.parameters['word_lstm_dim'],
                           use_gpu=self.use_gpu,
                           char_to_ix=char_to_id,
                           pre_word_embeds=word_embeds,
                           use_crf=self.parameters['crf'],
                           char_mode=self.parameters['char_mode'])

        try:
            self.model.load_state_dict(torch.load(self.model_name))
        except:
            print("No model available!!!")

        if self.use_gpu:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.tag_to_id = tag_to_id
        self.id_to_tag = id_to_tag

    def train(self):
        losses = []
        loss = 0.0
        best_dev_F = -1.0
        best_test_F = -1.0
        best_train_F = -1.0
        all_F = [[0, 0, 0]]
        plot_every = 10
        eval_every = 20
        count = 0
        vis = visdom.Visdom()
        sys.stdout.flush()
        t = time.time()
        self.model.train(True)
        for epoch in range(1, 10001):
            for i, index in enumerate(np.random.permutation(len(self.train_data))):
                tr = time.time()
                count += 1
                data = self.train_data[index]
                self.model.zero_grad()

                sentence_in = data['words']
                sentence_in = Variable(torch.LongTensor(sentence_in))
                tags = data['tags']
                chars2 = data['chars']

                ######### char lstm
                if self.parameters['char_mode'] == 'LSTM':
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
                if self.parameters['char_mode'] == 'CNN':
                    d = {}
                    chars2_length = [len(c) for c in chars2]
                    char_maxl = max(chars2_length)
                    chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
                    for i, c in enumerate(chars2):
                        chars2_mask[i, :chars2_length[i]] = c
                    chars2_mask = Variable(torch.LongTensor(chars2_mask))


                targets = torch.LongTensor(tags)
                caps = Variable(torch.LongTensor(data['caps']))
                if self.use_gpu:
                    neg_log_likelihood = self.model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), caps.cuda(), chars2_length, d)
                else:
                    neg_log_likelihood = self.model.neg_log_likelihood(sentence_in, targets, chars2_mask, caps, chars2_length, d)
                loss += neg_log_likelihood.data[0] / len(data['words'])
                neg_log_likelihood.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                self.optimizer.step()

                if count % plot_every == 0:
                    loss /= plot_every
                    print(count, ': ', loss)
                    if losses == []:
                        losses.append(loss)
                    losses.append(loss)
                    text = '<p>' + '</p><p>'.join([str(l) for l in losses[-9:]]) + '</p>'
                    losswin = 'loss_' + self.parameters['name']
                    textwin = 'loss_text_' + self.parameters['name']
                    vis.line(np.array(losses), X=np.array([plot_every*i for i in range(len(losses))]),
                        win=losswin, opts={'title': losswin, 'legend': ['loss']})
                    vis.text(text, win=textwin, opts={'title': textwin})
                    loss = 0.0

                if count % (eval_every) == 0 and count > (eval_every * 20) or \
                        count % (eval_every*4) == 0 and count < (eval_every * 20):
                    self.model.train(False)
                    best_train_F, new_train_F, _ = self.evaluating(self.model, self.train_data, best_train_F)
                    best_dev_F, new_dev_F, save = self.evaluating(self.model, self.dev_data, best_dev_F)
                    if save:
                        torch.save(model, self.model_name)
                    best_test_F, new_test_F, _ = self.evaluating(self.model, self.test_data, best_test_F)
                    sys.stdout.flush()

                    all_F.append([new_train_F, new_dev_F, new_test_F])
                    Fwin = 'F-score of {train, dev, test}_' + self.parameters['name']
                    vis.line(np.array(all_F), win=Fwin,
                        X=np.array([eval_every*i for i in range(len(all_F))]),
                        opts={'title': Fwin, 'legend': ['train', 'dev', 'test']})
                    self.model.train(True)

                if count % len(self.train_data) == 0:
                    adjust_learning_rate(self.optimizer, lr=self.learning_rate/(1+0.05*count/len(self.train_data)))
                    
        print(time.time() - t)
        plt.plot(losses)
        plt.show()

    def evaluating(self, model, datas, best_F):
        prediction = []
        save = False
        new_F = 0.0
        confusion_matrix = torch.zeros((len(self.tag_to_id) - 2, len(self.tag_to_id) - 2))
        for data in datas:
            ground_truth_id = data['tags']
            words = data['str_words']
            chars2 = data['chars']
            caps = data['caps']

            if self.parameters['char_mode'] == 'LSTM':
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

            if self.parameters['char_mode'] == 'CNN':
                d = {}
                chars2_length = [len(c) for c in chars2]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
                for i, c in enumerate(chars2):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))

            dwords = Variable(torch.LongTensor(data['words']))
            dcaps = Variable(torch.LongTensor(caps))
            if self.use_gpu:
                val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, d)
            else:
                val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
            predicted_id = out
            for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
                line = ' '.join([word, self.id_to_tag[true_id], self.id_to_tag[pred_id]])
                prediction.append(line)
                confusion_matrix[true_id, pred_id] += 1
            prediction.append('')
        predf = eval_temp + '/pred.' + self.parameters['name']
        scoref = eval_temp + '/score.' + self.parameters['name']

        with open(predf, 'wb') as f:
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
            *([self.id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
        ))
        for i in range(confusion_matrix.size(0)):
            print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
                str(i), self.id_to_tag[i], str(confusion_matrix[i].sum()),
                *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
                  ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
            ))
        return best_F, new_F, save

if __name__ == '__main__':
    conf = Config
    train_pipeline = Trainpipeline(conf)
    train_pipeline.load()
    train_pipeline.train()
