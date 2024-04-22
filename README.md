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

## Necessary datasets  

#### Natural language inference (NLI) data

Perform the following steps to obtain the data used for training the sentence encoders:

1. Create a data folder from the root directory of the repository.
2. Download the dataset: wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip -P ./data
3. Unzip the dataset in data/ folder: unzip ./data/snli_1.0.zip -d ./data

### Transfer task data

The datasets for the transfer tasks can be obtained by running in SentEval/data/downstream/:
./get_transfer_data.bash

## Model requirements

### Glove embeddings
Perform the following steps to obtain the glove embeddings. 

1. wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P ./src/SentEval/pretrained
2. unzip src/SentEval/pretrainedglove.840B.300d.zip -d ./src/SentEval/pretrained
3. All files import glove file from **pretrained**/ folder in SentEval


### Pre-trained models


Download **weights** folder with pretrained models here: https://drive.google.com/drive/folders/1tv-pQ7J2LAA2HXf6uThkaN8JbMa9JDIu?usp=sharing

Folder contains weights **and** tensorboard files of training runs for the following models: 

1. 'bow': for the bag of words model, where glove embeddings of the sequence are averaged to obtain sentence representations.

2. 'lstm': for the LSTM model where the final hidden state is used as a sentence representation.

3. 'lstm_max': for the LSTM model with max pooling over the hidden states of the sequence (for extra research question). 

4. 'bilstm': for the BiLSTM model where the concatenation of the final hidden states (of the forward and backward LSTM) is used as sentence representation.

5. 'bilstm_max': the BiLSTM model with max pooling over the concatenated hidden states (of forward and backward LSTM) of the sequence. 

Move weights folder in src/



## Code structure

Src/ contains the following files for training and evaluating the models: 
1. preprocess.py: contains all code for preprocessing and creating the datasets. Train.py and eval.py import functions from here
2. train.py: for training the models with NLI
3. eval.py: for evaluating the models on transfer tasks + dev/test set of NLI
4. utils.py: extra functions such as for loading the models, computing metrics and saving/loading json files
5. demo.ipynb: for experimenting with the pre-trained models: new samples, extracting predictions on test set and experiment with varying input length 

### train.py 

The training of a model can be initialized with: 

python -u train.py --model <model> --checkpoint_path <chekpoint_path> 

Example: python -u train.py --model bilstm_max --checkpoint_path weights/bilstm/bilstm_max_best.pth

--**model** specifies which model to train: 
Options: bow, lstm, bilstm (concatenation of final hidden states), bilstm_max (max pooling over the concatentation of hidden states)

optional: 

--checkpoint_path: initializes model from given checkpoint

--batch_size: batch_size for training and evaluation, default: 64

--lr: starting learning rate, default: 0.1

--seed: default: 42

### Eval.py
The evaluation of a model can be initialized with: 

python -u eval.py --model <model> --checkpoint_path <chekpoint_path> 

Example: python -u eval.py --model bilstm_max --checkpoint_path weights/bilstm/bilstm_max_best.pth

--**model** specifies which model to evaluate: 
Options: bow, lstm, bilstm, bilstm_max. Default: bow

optional: 

--**eval_nli**: set to True if evaluation of dev and test set of NLI is desired. Default: false

--**checkpoint_path**: initializes model from given checkpoint

--batch_size: batch_size for training and evaluation, default: 64

--seed: default: 42

--device: evaluation on transfer tasks can be done on cpu. Eval_nli is always on gpu if available. Default: cpu








