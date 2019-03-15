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

## Requirement
- Tensorflow
- pynlpir
- jieba

## Usage
Note: Current repo only contains *Version 1* which is discussed in **Experiments**, but you could easily modify the cofigurations in *main.py* to have the *Baseline* set in **Experiments**
1. Preprocess the dataset, whose format should be the same as the examples in */data*. And the code in preprocess.py shall be considered as the refernce.
2. Yield a character-level embedding with Glove or fastText. Refer to the *README.md* in */Glove* and */fastText* to gain an intuition.
3. Modify configurations in *main.py* to get a brand new config suiting your idea.
4. Train your own model by  
```bash
python main.py --train=True
```
5. Evaluate the outcome of your model by  
```bash
python main.py
```

## Contributions

If you have any question about this repo or you find some problems in my code, feel free to contact with me via email or just give comments in *Issues*.

## Experiments

### Baseline
- Corpus: MSRA

- Tag schema: iobes
  > 我 S-PER  
  > 来 O  
  > 自 O  
  > 北 B-ORG  
  > 京 I-ORG  
  > 理 I-ORG  
  > 工 I-ORG  
  > 大 I-ORG  
  > 学 E-ORG  

- Dataset split
  > 90% train, 10% test 

- Structure
  > - with pretrained GloVe embedding for characters and random initialized embedding for segmentations
  > - concatenate above two
  > - Bi-LSTM
  > - CRF loss layer

- Result(Test only)

  | Type | Accuracy | Precision | Recall | FB1 |
  | :-: | :-: | :-: | :-: | :-: |
  | LOC | \ | 94.11% | 90.58% | 92.31 |
  | ORG | \ | 86.86% | 88.43% | 87.64 |
  | PER | \ | 92.21% | 92.45% | 92.33 |
  | OVER ALL | 98.79% | 91.89% | 90.71% | 91.30 |

### Version 1 (Bi-LSTM + CRF + CHAR_CNN)
- Corpus: SAME as baseline

- Tag schema: iobes
  > 我 S-PER  
  > 来自 O  
  > 北京 B-ORG  
  > 理工 I-ORG  
  > 大学 E-ORG

- Dataset split: SAME as baseline

- Structure
  > - with pretrained GloVe embedding for words
  > - use a layer of CNN to capture the features of characters projected by embedding
  > - Bi-LSTM
  > - CRF loss layer
  
- Result(Test only)

  | Type | Accuracy | Precision | Recall | FB1 |
  | :-: | :-: | :-: | :-: | :-: |
  | LOC | \ | 93.31% | 91.14% | 92.21 |
  | ORG | \ | 87.15% | 86.63% | 86.89 |
  | PER | \ | 93.94% | 91.94% | 92.93 |
  | OVER ALL | 98.83% | 92.17% | 90.42% | 91.29 |
