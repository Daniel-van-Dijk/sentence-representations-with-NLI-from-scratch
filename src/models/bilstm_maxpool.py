import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM_MaxPool_NLI(nn.Module):

    def __init__(self, embedding_dim, bilstm_dim, hidden_dim, vocab_size, num_labels):
        super(BiLSTM_MaxPool_NLI, self).__init__()
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
        # pack padded variable length input
        packed1 = pack_padded_sequence(embeds1, lengths1, batch_first=True, enforce_sorted=False)
        packed2 = pack_padded_sequence(embeds2, lengths2, batch_first=True, enforce_sorted=False)

        # output contains concatenated hidden states of bilstm, but needs unpacking
        output1, (_, _) = self.bilstm(packed1)
        output2, (_, _) = self.bilstm(packed2)

        # do the unpacking to batch size x max_length x embedding_dim*2 
        unpacked1, _ = pad_packed_sequence(output1, batch_first=True)
        unpacked2, _ = pad_packed_sequence(output2, batch_first=True)

        # take maximum over sequence dim -> batch_size x embedding_dim * 2
        u = torch.max(unpacked1, dim=1)[0]
        v = torch.max(unpacked2, dim=1)[0]
        diff = torch.abs(u - v)
        dotprod = u * v
        combined = torch.hstack([u, v, diff, dotprod])
        logits = self.mlp(combined)
        output = F.softmax(logits, dim=1)
        return output