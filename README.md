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
  
