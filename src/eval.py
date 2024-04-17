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
from utils import *

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


def prepare(params, samples):
    return

# # SentEval prepare and batcher
# def prepare(params, samples):
#     train_split = preprocess(split='train')
#     #vocab = create_vocab(train_split)
#     #embeddings = align_vocab_with_glove(vocab)
#     params.vocab = create_vocab(train_split)
#     params.embeddings = align_vocab_with_glove(params.vocab)
#     #_, params.word2id = create_dictionary(samples)
#     #params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
#     params.wvec_dim = 300
#     return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    #print('batch size', len(batch))
    embeddings_list = []
    for sent in batch:
        sent_vec = []
        for token in sent:
            # get token ID's based on NLI vocab, unknown to 0 (<unk>)
            sent_vec.append(params.vocab.mapping.get(token, 0))
        # get glove embeddings of sentence based on token ID's
        embeddings = params.bow.token_embeddings(torch.LongTensor([sent_vec]))
        # !! only for bow
        embeddings_list.append(embeddings.mean(1))
    # stack sentence representations of batch -> batch size x dim
    embeddings_list = torch.vstack(embeddings_list)
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

    train_split = preprocess(split='dev')
    vocab = create_vocab(train_split)
    embeddings = align_vocab_with_glove(vocab)
    labels = ['neutral', 'entailment', 'contradiction']
    params_senteval['vocab'] = vocab
    params_senteval['embeddings'] = embeddings
    vocab_size = len(vocab.mapping)

    model = load_model(embeddings, labels, vocab_size, device, args.model, args.checkpoint_path)

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
