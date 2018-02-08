## Chinese-NER
An attempt to do better on Chinese NER task.
## Reference & Inspiration
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
  > - with pretrained glove embedding
  > - Bi-LSTM (word-level input and character-level output)
  > - Bi-LSTM (character-level input)
  > - concat above two to form the real input 
  > - CRF loss layer
  > - LSTM (capture the features of Capitalization)
- [Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition](http://tcci.ccf.org.cn/conference/2016/papers/119.pdf)
  > - with pretrained word2vec(character2vec may be more proper) embedding
  > - Bi-LSTM (character-level input)
  > - Bi-LSTM (radical-level input)
  > - concat above two
  > - CRF loss layer
- [A neural network model for Chinese named entity recognition](https://github.com/zjy-ucas/ChineseNER)
  > - with pretrained word2vec(same as above one) embedding
  > - Bi-LSTM (character-level inut)
  > - Bi-LSTM or CNN(CNN is better) (do a segmentation operation to yield input)
  > - concat above two
  > - CRF loss layer
- [NER-pytorch](https://github.com/ZhixiuYe/NER-pytorch)
  > - a pytorch version implementation which is similar to the very beginning one
## Some Intuitions
- replace word2vec by [GloVe](https://github.com/stanfordnlp/GloVe)
  > that is, '北理工' will be trained rather than '北'、'理'、'工' be separately pre-trained
- using [fastText](https://github.com/facebookresearch/fastText)'s ideas: 
  > so called sub-words, and not in character-level, in word-level instead.
  > in Chinese, means different combinations of characters
  > for example, in a word embedding '北理工', \['北','理','工','北理','理工'\] will be considered differently
- or, using CNN layer to capture features above
- futhermore, I found that simplified version of organization or localization names are hard to capture
  > it might be the problem of pretrained corpus
  > or we should come up with a solution to solve this
