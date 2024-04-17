# Reproducing "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data" by Conneau et al. ([https://arxiv.org/abs/2203.15395](https://arxiv.org/abs/1705.02364)) from scratch

## Set up

### Dependencies
```
- Python 3.9
- numpy 1.24.2 
- torch 2.2.2 +cu118
- tensorboard 2.16.2
- spacy 3.7.4
- scikit-learn 1.4.2 
```
The environment can be installed with: "conda env create -f env.yml"

### Data

#### Natural language inference (NLI) data

Perform the following steps to obtain the data used for training the sentence encoders:

1. Create a data folder from the root directory of the repository.
2. Download the dataset: wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip -P ./data
3. Unzip the dataset in data/ folder: unzip ./data/snli_1.0.zip -d ./data

### Transfer task data

The datasets for the transfer tasks can be obtained by running in SentEval/data/downstream/:
./get_transfer_data.bash

### Glove embeddings
Perform the following steps to obtain the glove embeddings. 

1. wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P ./src/SentEval/pretrained
2. unzip src/SentEval/pretrainedglove.840B.300d.zip -d ./src/SentEval/pretrained
3. All files import glove from pretrained folder in SentEval

## code structure

## train.py 

The training is done with train.py

python -u train.py --model bilstm_max
--model specifies which model to evaluate: 
Options: bow, lstm, bilstm, bilstm_max

### Eval.py
The evaluation is done with eval.py, 

python -u eval.py --model bilstm --checkpoint_path weights/bilstm/bilstm_max_best.pth --device cpu

--model specifies which model to evaluate: 
Options: bow, lstm, bilstm, bilstm_max









