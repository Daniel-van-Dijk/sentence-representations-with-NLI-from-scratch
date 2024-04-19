import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess import *
from utils import *
from eval import *
import os
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from models.bow import BOW
from models.lstm import LSTM_NLI
from models.bilstm import BiLSTM_NLI
from models.bilstm_maxpool import BiLSTM_MaxPool_NLI
import datetime
import argparse
# for logging
TIME = datetime.datetime.now()
import copy


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

def train(train_loader, validation_loader, model, loss_module, device, num_epochs=50, model_flag='bow'):
    writer = SummaryWriter(f'runs/{model_flag}/{TIME}')
    weights_folder = f'weights/{model_flag}'
    if not os.path.exists(weights_folder):
         os.makedirs(weights_folder)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # learning rate * 0.99 after every epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    last_validation_accuracy = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        print(f'starting epoch {epoch} with learning rate: {scheduler.get_last_lr()[0]}')
        for i, (sent1s, sent2s, labels, lengths1, lengths2) in enumerate(train_loader, 0):
            sent1s = sent1s.to(device)
            sent2s = sent2s.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if model_flag == 'bow':
                output = model(sent1s,sent2s)

            elif model_flag == 'lstm' or model_flag == 'bilstm' or model_flag == 'bilstm_max':
                output = model(sent1s,lengths1, sent2s, lengths2)
            output_loss = loss_module(output, labels)
            train_loss += output_loss.item()
            output_loss.backward()
            optimizer.step()

        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_module, device, model_flag)
        
        # saving model with lowest validation loss
        if validation_loss < best_val_loss:
            print(f'New best validation_loss: {validation_loss}')
            print(f'Saving current best model with accuracy {validation_accuracy}')
            best_val_loss = validation_loss
            # deep copy to not affect training loop
            sd = copy.deepcopy(model.state_dict())
            del sd['token_embeddings.weight']
            torch.save(sd, f'{weights_folder}/{model_flag}_best_{TIME}.pth')

        avg_train_loss = train_loss / len(train_loader)
        print(f'avg training loss at epoch {epoch}: {avg_train_loss}')
        print(f'avg validation loss at epoch {epoch}: {validation_loss}')
        print(f'Validation accuracy at epoch {epoch}: {validation_accuracy }%')
        # tensorboard logging
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", validation_loss, epoch)
        writer.add_scalar("Accuracy/validation", validation_accuracy, epoch)
        if validation_accuracy < last_validation_accuracy:
            print('accuracy decreased')
            for group in optimizer.param_groups:
                # decrease lr by 5 if validation accuracy decreases
                group['lr'] /= 5
        last_validation_accuracy = validation_accuracy
        # update LR
        scheduler.step()
        if scheduler.get_last_lr()[0] < 0.00001:
            print('reached treshold, stopping training')
            writer.flush()
            writer.close()
            sd = model.state_dict()
            del sd['token_embeddings.weight']
            torch.save(sd, f'{weights_folder}/{model_flag}_{TIME}.pth')
            break
    # also save if max epochs reached instead of lr below treshold
    sd = model.state_dict()
    # do not save embeddings
    del sd['token_embeddings.weight']
    torch.save(sd, f'{weights_folder}/{model_flag}_{TIME}.pth')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    print(device)
    labels = ['neutral', 'entailment', 'contradiction']
    # map label to numeric
    label_mapping = {label: index for index, label in enumerate(labels)}

    train_split = preprocess(split='train')
    vocab_file = 'NLI_vocab.pkl'
    folder = 'saved_files'
    if os.path.exists(f'{folder}/{vocab_file}'):
        print('loading vocab from file')
        vocab = load_file(vocab_file)
    else:
        print('creating vocab with training set')
        vocab = create_vocab(train_split)
        save_file(vocab, vocab_file)
    embeddings = align_vocab_with_glove(vocab)

    # TextpairDataset and custom_collate function in preprocess.py
    train_data = TextpairDataset(train_split, vocab, label_mapping)
    print(f'Train set size: {len(train_data)}')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    print(f'Number of train batches: {len(train_loader)}')

    validation_split = preprocess(split='dev')
    validation_data = TextpairDataset(validation_split, vocab, label_mapping)
    print(f'Validation set size: {len(validation_data)}')

    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    vocab_size = len(vocab.mapping)
    print(f'size of vocab: {vocab_size}')
    loss_module = nn.CrossEntropyLoss()
    model = load_model(embeddings, labels, vocab_size, device, args.model, args.checkpoint_path)
    train(train_loader, validation_loader, model, loss_module, device, model_flag=args.model)

if __name__ == '__main__':
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
    main(args)