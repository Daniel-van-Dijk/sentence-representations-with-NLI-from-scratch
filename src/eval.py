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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    parser = argparse.ArgumentParser('Evaluating sentence representations', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training and evaluation')
    parser.add_argument('--model', default='bow', type=str, metavar='MODEL',
                        help='model to train')
    parser.add_argument('--checkpoint_path', type=str,
                        help='checkpoint path')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--eval_nli', default=False, type=bool)
    return parser

def evaluate(model, dataloader, loss_module, device, model_flag):
    """ 
    For evaluating models on NLI dev and test set, also used in train.py
    Returns loss and accuracy
    """
    model.eval()
    val_loss = 0
    total_preds, correct_preds = 0, 0
    with torch.no_grad():
        for i, (sent1s, sent2s, labels, lengths1, lengths2) in enumerate(dataloader, 0):
            sent1s = sent1s.to(device)
            sent2s = sent2s.to(device)
            labels = labels.to(device)
            if model_flag == 'bow':
                output = model(sent1s,sent2s)

            elif model_flag == 'lstm' or model_flag == 'bilstm' or model_flag == 'bilstm_max':
                output = model(sent1s,lengths1, sent2s, lengths2)
            predicted_labels = torch.argmax(output, dim=1)
            correct_preds += (predicted_labels == labels).sum() 
            total_preds += labels.shape[0]
            output_val_loss = loss_module(output, labels)
            val_loss += output_val_loss.item()
    return val_loss / len(dataloader), correct_preds / total_preds

def prepare(params, samples):
    """
    Prepare function is not used since vocabulary of NLI is taken
    """
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    max_length = max(len(sent) for sent in batch)
    padding_ID = 1
    sents, lengths = [], []
    for sent in batch:
        sent_vec = []
        for token in sent:
            # get token ID's based on NLI vocab, unknown to 0 (<unk>)
            sent_vec.append(params.vocab.mapping.get(token, 0))
        # pad sentence to max length of batch
        padded_sent = sent_vec + [padding_ID] * (max_length - len(sent))
        sents.append(padded_sent), lengths.append(len(sent))
    padded_batch = torch.tensor(sents)
    padded_batch = padded_batch.to(params.device)
    # get glove embeddings of sentence based on token ID's
    embeddings = params.model.token_embeddings(padded_batch)

    if params['model_flag'] == 'bow':
        representations = embeddings.mean(1)
        return representations.detach().numpy()

    elif params['model_flag'] == 'lstm':
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        _, (u, _) = params.model.lstm(packed)
        representations = u.squeeze(0)
        # avoid error:  Can't call numpy() on Tensor that requires grad
        return representations.detach().numpy()
    
    elif params['model_flag'] == 'bilstm':
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        _, (u, _) = params.model.bilstm(packed)
        u = u.transpose(0, 1)
        representations = u.reshape(u.shape[0], -1)
        return representations.detach().numpy()
    
    elif params['model_flag'] == 'bilstm_max':
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        output, (_, _) = params.model.bilstm(packed)
        unpacked, _ = pad_packed_sequence(output, batch_first=True)
        representations = torch.max(unpacked, dim=1)[0]
        return representations.detach().numpy()

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
# params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                  'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    device = args.device
    print('device', device)
    print(f'seed: {args.seed}')
    print(f'model: {args.model}')
    if args.checkpoint_path:
        print(f'loading checkpoint from {args.checkpoint_path}')
    else:
        print('no checkpoint path provided')
        print(args.checkpoint_path)

    train_split = preprocess(split='train')
    vocab = create_vocab(train_split)
    embeddings = align_vocab_with_glove(vocab)
    labels = ['neutral', 'entailment', 'contradiction']
    params_senteval['vocab'] = vocab
    params_senteval['embeddings'] = embeddings
    params_senteval['device'] = device
    vocab_size = len(vocab.mapping)

    model = load_model(embeddings, labels, vocab_size, device, args.model, args.checkpoint_path)
    

    if args.eval_nli:
        print(f'Evaluating {args.model} on NLI dev and test set')
        # always try to evaluate NLI on gpu 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_module = nn.CrossEntropyLoss()

        torch.manual_seed(args.seed)
        model_nli = load_model(embeddings, labels, vocab_size, device, args.model, args.checkpoint_path)
        label_mapping = {label: index for index, label in enumerate(labels)}

        validation_split = preprocess(split='dev')
        validation_data = TextpairDataset(validation_split, vocab, label_mapping)
        print(f'Validation set size: {len(validation_data)}')
        validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, 
                                                        collate_fn=custom_collate)
        _, validation_accuracy = evaluate(model_nli, validation_loader, loss_module, device, args.model)
        
        test_split = preprocess(split='test')
        test_data = TextpairDataset(test_split, vocab, label_mapping)
        print(f'Test set size: {len(test_data)}')
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, 
                                                        collate_fn=custom_collate)
        _, test_accuracy = evaluate(model_nli, test_loader, loss_module, device, args.model)

        print("Validation accuracy on NLI:", validation_accuracy )
        print("Test accuracy on NLI:", test_accuracy)

        del model_nli



    params_senteval['model'] = model
    params_senteval['model_flag'] = args.model
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'STS14', 'SICKRelatedness']

    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    print(results)

    print('================================================================')
    print(f"Calculating micro and macro average for {args.model}")
    print("\n")

    micro, macro = get_metrics(results)
    
    print('macro accuracy', macro)
    print('micro accuracy', micro)

    print('============')
    print("STS14 results")
    print(results['STS14']['all'])
    print('----')
    print('============')
    print("SICKRelatedness results")
    print(results['SICKRelatedness'])
