# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import argparse

from preprocess import *
from models.bow import BOW
from models.lstm import LSTM_NLI
from models.bilstm import BiLSTM_NLI
from models.bilstm_maxpool import BiLSTM_MaxPool_NLI


# Set PATHs
PATH_TO_SENTEVAL = './SentEval/'
PATH_TO_DATA = './SentEval/data/'
PATH_TO_VEC = './SentEval/pretrained/glove.840B.300d.txt'

# Add SentEval to the system path
sys.path.insert(0, PATH_TO_SENTEVAL)

# Import SentEval
import senteval

def get_args_parser():
    parser = argparse.ArgumentParser('Learning sentence representations via NLI', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training and evaluation')
    parser.add_argument('--model', default='bow', type=str, metavar='MODEL',
                        help='model to train')
    parser.add_argument('--checkpoint_path', type=str,
                        help='checkpoint path')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--seed', default=42, type=int)
    return parser

def create_vocab(dataset):
  v = Vocabulary()
  for sent in dataset:
    for token in sent:
      v.add_token(token)
  v.create_mapping()
  return v

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    train_split = preprocess(split='train')
    vocab = create_vocab(train_split)
    embeddings = align_vocab_with_glove(vocab)
    params.vocab = create_vocab(samples)
    params.embeddings = align_vocab_with_glove(params.vocab)
    #_, params.word2id = create_dictionary(samples)
    #params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    return

def batcher(params, batch):
    embeddings_list = []
    for sent in batch:
        sent_vec = []
        for token in sent:
            sent_vec.append(params.vocab.mapping.get(token, 0))
        embeddings = params.bow.token_embeddings(sent_vec)
        embeddings_list.append(np.mean(embeddings))

    embeddings_list = np.vstack(embeddings_list)
    print(embeddings.shape)
    x = x - 3
    return embeddings

def batcher2(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    print('length of batch', len(batch))
    print(batch)

    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.vocab:
                id = params.vocab.mapping.get(word, 0)
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    print(embeddings.shape)
    x = x - 3
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 2}
# params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                  'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    args = get_args_parser()
    args = args.parse_args()
    print(f'seed: {args.seed}')
    print(f'learning rate:  {args.lr}')
    print(f'model: {args.model}')
    if args.checkpoint_path:
        print(f'loading checkpoint from {args.checkpoint_path}')
    else:
        print('no checkpoint path provided')
        print(args.checkpoint_path)
    vocab_size = 33623
    model = BOW(300, 512, 33623, 3)
    state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)

    params_senteval['bow'] = model.to(device)
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']
    
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                     'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                 'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                 'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                 'OddManOut', 'CoordinationInversion']

     # here you define the NLP taks that your embedding model is going to be evaluated
    # in (https://arxiv.org/abs/1802.05883) we use the following :
    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression), 
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    #transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
    #                  'MRPC', 'SICKEntailment', 'STS14']
    transfer_tasks = ['MR', 'CR']
    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    print(results)
