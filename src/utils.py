import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess import *
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


def load_model(embeddings, labels, vocab_size, device, model_flag='lstm', state_file_path = None):
    embedding_dim = 300
    hidden_dim = 512
    print(f'loading {model_flag}')
    if model_flag == 'lstm':
        lstm_dim = 2048
        model = LSTM_NLI(embedding_dim, lstm_dim, hidden_dim, vocab_size, len(labels))
        
    
    elif model_flag == 'bilstm':
        bilstm_dim = 2048
        model = BiLSTM_NLI(embedding_dim, bilstm_dim, hidden_dim, vocab_size, len(labels))
    
    elif model_flag == 'bilstm_max':
        bilstm_dim = 2048
        model = BiLSTM_MaxPool_NLI(embedding_dim, bilstm_dim, hidden_dim, vocab_size, len(labels))

        
    elif model_flag == 'bow':
        model = BOW(embedding_dim, hidden_dim, vocab_size, len(labels) )
       

    model.token_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
    #keep embeddings fixed during training
    model.token_embeddings.weight.requires_grad = False
    if state_file_path:
        if str(device) == 'cpu':
            state_dict = torch.load(state_file_path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(state_file_path)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    return model