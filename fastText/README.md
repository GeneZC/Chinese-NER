## Usage
```bash
$ make
$ ./fasttext skipgram -input text8 -output model
```
At the end of optimization the program will save two files: model.bin and model.vec. model.vec is a text file containing the word vectors, one per line. model.bin is a binary file containing the parameters of the model along with the dictionary and all hyper parameters. The binary file can be used later to compute word vectors or to restart the optimization.
## Tips
- You could edit **src/fasttext.cpp** to change for parameters you prefer
- You could replace **text8**, which is the corpus. But do notice the file format.
