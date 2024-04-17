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
3. All files import glove file from pretrained/ folder in SentEval


### Pre-trained models

Download weights folder with pretrained models here: https://drive.google.com/drive/folders/1tv-pQ7J2LAA2HXf6uThkaN8JbMa9JDIu?usp=sharing
Move weights folder in src/

## Code structure

Src/ contains the following files for training and evaluating the models: 
1. preprocess.py: contains all code for preprocessing and creating the datasets. Train.py and eval.py import functions from here
2. train.py: for training the models with NLI
3. eval.py: for evaluating the models
4. utils.py: extra functionalities such as code for loading the models. 

### train.py 

The training of a model can be initialized with: 

python -u train.py --model <model> --checkpoint_path <chekpoint_path> 
Example: python -u train.py --model bilstm_max --checkpoint_path weights/bilstm/bilstm_max_best.pth

--model specifies which model to train: 
Options: bow, lstm, bilstm, bilstm_max

optional: 
--checkpoint_path: initializes model from given checkpoint
--batch_size: batch_size for training and evaluation, default: 64
--lr: starting learning rate, default: 0.1
--seed: default: 42

### Eval.py
The evaluation of a model can be initialized with: 

python -u eval.py --model <model> --checkpoint_path <chekpoint_path> 
Example: python -u eval.py --model bilstm_max --checkpoint_path weights/bilstm/bilstm_max_best.pth

--model specifies which model to evaluate: 
Options: bow, lstm, bilstm, bilstm_max

optional: 
--checkpoint_path: initializes model from given checkpoint








