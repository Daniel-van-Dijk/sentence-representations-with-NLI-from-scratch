import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM_NLI(nn.Module):

    def __init__(self, embedding_dim, lstm_dim, hidden_dim, vocab_size, num_labels):
        super(LSTM_NLI, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_dim, batch_first=True)
        # dim * 4 as we concatenate 4 vectors: u, v, |u - v|, u*v
        #self.linear1 = nn.Linear(lstm_dim*4, hidden_dim)
        #self.linear2 = nn.Linear(hidden_dim, num_labels)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_dim*4, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, num_labels),
            )

    def forward(self, sentence1, lengths1, sentence2, lengths2):
        embeds1 = self.token_embeddings(sentence1)
        embeds2 = self.token_embeddings(sentence2)
        # pack padded variable input
        packed1 = pack_padded_sequence(embeds1, lengths1, batch_first=True, enforce_sorted=False)
        packed2 = pack_padded_sequence(embeds2, lengths2, batch_first=True, enforce_sorted=False)

        _, (u, _) = self.lstm(packed1)
        _, (v, _) = self.lstm(packed2)
        # [1, 64, 300] -> [64, 300]
        u = u.squeeze(0)
        v = v.squeeze(0)

        diff = torch.abs(u - v)
        dotprod = u * v
        combined = torch.hstack([u, v, diff, dotprod])
        logits = self.mlp(combined)
        output = F.softmax(logits, dim=1)
        return output