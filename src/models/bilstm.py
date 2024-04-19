import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class BiLSTM_NLI(nn.Module):

    def __init__(self, embedding_dim, bilstm_dim, hidden_dim, vocab_size, num_labels):
        super(BiLSTM_NLI, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=bilstm_dim, batch_first=True, bidirectional=True)
        # dim * 8 as we concatenate 4 concatenated 4096-dim (from 2 directions) vectors : u, v, |u - v|, u*v
        self.mlp = nn.Sequential(
            nn.Linear(bilstm_dim*8, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, num_labels),
            )

    def forward(self, sentence1, lengths1, sentence2, lengths2):
        embeds1 = self.token_embeddings(sentence1)
        embeds2 = self.token_embeddings(sentence2)
        # pack padded variable input
        packed1 = pack_padded_sequence(embeds1, lengths1, batch_first=True, enforce_sorted=False)
        packed2 = pack_padded_sequence(embeds2, lengths2, batch_first=True, enforce_sorted=False)
        # use final hidden states  
        _, (u, _) = self.bilstm(packed1)
        _, (v, _) = self.bilstm(packed2)
        # [2, 64, 2048] -> [64, 2, 2048]
        u = u.transpose(0, 1)
        v = v.transpose(0, 1)
        # [64, 2, 2048] -> [64, 4096]
        u = u.reshape(u.shape[0], -1)
        v = v.reshape(v.shape[0], -1)
        diff = torch.abs(u - v)
        dotprod = u * v
        combined = torch.hstack([u, v, diff, dotprod])
        logits = self.mlp(combined)
        return logits