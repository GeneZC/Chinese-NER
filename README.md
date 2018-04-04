## Chinese-NER
An attempt to do better on Chinese NER task.
## Reference & Inspiration
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
  > - with pretrained glove embedding
  > - Bi-LSTM (character-level input)
  > - (optional) LSTM (capture the features of Capitalization)
  > - concatenate above two (or three) to form the real input
  > - Bi-LSTM (word-level input and character-level input) 
  > - CRF loss layer
- [Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition](http://tcci.ccf.org.cn/conference/2016/papers/119.pdf)
  > - with pretrained word2vec(character2vec may be more proper) embedding
  > - Bi-LSTM (radical-level input)
  > - concatenate above two
  > - Bi-LSTM (real input)
  > - CRF loss layer
- [A neural network model for Chinese named entity recognition](https://github.com/zjy-ucas/ChineseNER)  
See also in [Improving Named Entity Recognition for Chinese Social Media
with Word Segmentation Representation Learning](http://anthology.aclweb.org/P/P16/P16-2025.pdf)
  > - with pretrained word2vec(same as above one) embedding
  > - do a segmentation operation to yield input
  > - concatenate above two
  > - Bi-LSTM (real input)
  > - CRF loss layer
- [NER-pytorch](https://github.com/ZhixiuYe/NER-pytorch)
  > - a pytorch version implementation which is similar to the very beginning one.
- [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473.pdf)
  > - a framework using GAN + RL to solve sequence problem.
- [Adversarial Learning for Chinese NER from Crowd Annotations](https://arxiv.org/pdf/1801.05147.pdf)
  > - 2018 AAAI Alibaba
- [RL-GAN For NLP: 强化学习在生成对抗网络文本生成中扮演的角色](http://www.zhuanzhi.ai/document/004615a522841d224fffcbb3abcb8213)

## Some Intuitions
- '北理工' will be trained rather than '北'、'理'、'工' be separately pre-trained
- using [fastText](https://github.com/facebookresearch/fastText)'s ideas:
  > so called sub-words, and not in character-level, in word-level instead.  
  > in Chinese, means different combinations of characters.  
  > for example, in a word embedding '北理工', \['北','理','工','北理','理工'\] will be considered differently
- or, using CNN layer to capture features above
- futhermore, I found that simplified version of organization or localization names are hard to capture
  > such as '央视' which represents '中央电视台'.  
  > it might be the problem of pretrained corpus  
  > or we should come up with a solution to solve this

## Usage

1. Preprocess the dataset, whose format should be the same as the examples in /data. And the code in preprocess.py shall be considered as refernce.
2. Yield a embedding with Glove or fastText. Refer to the README in /Glove and /fastText to gain an intuition.
3. Modify config.py to get a brand new config suitting your idea.
4. Train your own model by  
```bash
python train.py
```
5. Evaluate the outcome of your model by  
```bash
python eval.py
```

## Contributions

If you have any question about this repo or there exists some bugs of my code, please feel free to contact with me via email or just give comments in Issues

## Experiments

### Baseline
- RAW Corpus: 人民日报199801-词性标注

  ![Haha](https://raw.githubusercontent.com/GeneZC/Chinese-NER/master/figures/raw.png)

- Preprocessed(with iob as tag schema) 

  > 我 O  
  > 来 O  
  > 自 O  
  > 北 B-ORG  
  > 京 I-ORG  
  > 理 I-ORG  
  > 工 I-ORG  
  > 大 I-ORG  
  > 学 I-ORG  

- Dataset split
  > 75% train, 25% test  
  > 14384 / 4799 sentences in train / test.  
  > tag schema: iobes

- Structure
  > - with pretrained GloVe embedding for characters and random initialized embedding for segmentations
  > - concatenate above two
  > - Bi-LSTM
  > - CRF loss layer

- Result(Test only)

| Type | Accuracy | Precision | Recall | FB1 |
| :-: | :-: | :-: | :-: | :-: |
| LOC | \ | 93.15% | 93.60% | 93.38 |
| ORG | \ | 91.45% | 91.89% | 91.67 |
| PER | \ | 96.91% | 96.61% | 96.76 |
| OVER ALL | 99.15% | 94.24% | 94.41% | 94.33 |

### Version 1 (Bi-LSTM + CRF + CHAR_CNN)
- RAW Corpus: 人民日报199801-词性标注

- Preprocessed(with iob as tag schema):

  > 我 O  
  > 来自 O  
  > 北京 B-ORG  
  > 理工 I-ORG  
  > 大学 I-ORG  

- Dataset split: SAME as baseline

- Structure
  > - with pretrained GloVe embedding for words and random initialized embedding for characters
  > - use a layer of maxpooled CNN to capture the features of characters projected by embedding
  > - concatenate word-level input and above one
  > - Bi-LSTM
  > - CRF loss layer
  
- Result(Test only)

| Type | Accuracy | Precision | Recall | FB1 |
| :-: | :-: | :-: | :-: | :-: |
| LOC | \ | 95.01% | 91.76% | 93.36 |
| ORG | \ | 88.76% | 89.45% | 89.10 |
| PER | \ | 97.14% | 96.19% | 96.66 |
| OVER ALL | 98.81% | 94.46% | 92.87% | 93.66 |

### Version 2 (fastText)
- SAME as Version 1 except word embedding is trained with fastText

- Result(Test only)

| Type | Accuracy | Precision | Recall | FB1 |
| :-: | :-: | :-: | :-: | :-: |
| LOC | \ | 94.34% | 92.16% | 93.24 |
| ORG | \ | 88.03% | 83.74% | 85.83 |
| PER | \ | 96.80% | 95.34% | 96.06 |
| OVER ALL | 99.08% | 94.02% | 91.64% | 92.81 |

### Version 3 (Bi-LSTM + CRF + CHAR_CNN)
- RAW Corpus: 人民日报1998-词性标注 and 人民日报2002-词性标注

- Preprocessed(with iob as tag schema):

  > 我 O  
  > 来自 O  
  > 北 B-ORG  
  > 京 I-ORG  
  > 理 I-ORG  
  > 工 I-ORG  
  > 大 I-ORG  
  > 学 I-ORG  

- Dataset split:
  > 90% train, 10% test  
  > ~400000 / ~40000 sentences in train / test.  
  > tag schema: iobeslr

- Structure
  > - with pretrained GloVe embedding for words and random initialized embedding for characters
  > - use a layer of maxpooled CNN to capture the features of characters projected by embedding
  > - concatenate word-level input and above one
  > - Bi-LSTM
  > - CRF loss layer
  
- Result(Test only)

| Type | Accuracy | Precision | Recall | FB1 |
| :-: | :-: | :-: | :-: | :-: |
| LOC | \ | 87.82% | 91.46% | 89.60 |
| ORG | \ | 92.45% | 94.27% | 93.35 |
| PER | \ | 92.27% | 92.97% | 92.62 |
| OVER ALL | 98.59% | 90.58% | 92.76% | 91.66 |

- Problem
  > L and R tags are not concerned with the words themsleves (even space), not intuitive enough  
  > L and R tags may be overlap

### Version 4 (SeqGAN)
- RAW Corpus: MSRA

- Preprocessed(with iob as tag schema): SAME as baseline

- Dataset split: SAME as baseline

- Structure
  > See SeqGAN

- Result(Test only)

It didn't work as expected, which could almost reach 80 F1 score, though, costing a much longer time...

