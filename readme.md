## Implementation of different Machine Learning models for Part-of-Speech tagging

### Prerequisites

Make sure you have installed all of the following prerequisites on your machine:
- python3
- tensorflow
- numpy

### Baseline Classifier
We used Perceptron as baseline classifier for our POS tagger. To use baseline
classifer run the following commands.

```
cd baseline/perceptron/
python3 main.py
```

To change the data for baseline classifier, go to the data directory on 
baseline/perceptron directory and replace the data files from it with the
same name.

### Download Pre-trained Models and Embeddings
Rest of the models requires pre-trained embeddings to run and pre-trained
models if you don't want to train again. You can download the models and
embeddings from the following URL:
```buildoutcfg
https://drive.google.com/drive/folders/1lpW5pN9L34lMwePs1lug5PBzyiL6BI12?usp=sharing
```

For embeddings download the embeddings.zip from the URL and place the extracted 
file inside embeddings folder in root directory of the project. So that, the path
of the two embedding files in relative to project root would be as follows:
```buildoutcfg
{PROJECT_ROOT}/embeddings/pretrained/glove.6B.300d.txt
{PROJECT_ROOT}/embeddings/pretrained/wiki-news-300d-1M.vec
```

For the Pre-trained model download the two zip files 'rnn_fasttext_model'
and 'rnn_glove_model' and extract it on the project root.
```buildoutcfg
{PROJECT_ROOT}/rnn_fasttext_model/
{PROJECT_ROOT}/rnn_glove_model/
```

### Run Different Models
You can run the following commands on project root to run different models.
If you want to change the test or training data before running the models
 replace the data files inside data directory in project root. 

For Perceptron with GloVe Embeddings:
```buildoutcfg
python3 main.py --model=perceptron
```

For RNN with GloVe Embeddings run the following command:
```buildoutcfg
python3 main.py --model=rnn --embeddings=GLOVE --train=false
```
Change the train parameter to false if you want to train again.

For RNN with FASTTEXT embeddings run the following command:
```buildoutcfg
python3 main.py --model=rnn --embeddings=FASTTEXT --train=false
```