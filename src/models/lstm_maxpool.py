import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM_MaxPool_NLI(nn.Module):

    def __init__(self, embedding_dim, lstm_dim, hidden_dim, vocab_size, num_labels):
        super(LSTM_MaxPool_NLI, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_dim, batch_first=True)
        # dim * 4 as we concatenate 4 vectors: u, v, |u - v|, u*v
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

        output1, (_, _) = self.lstm(packed1)
        output2, (_, _) = self.lstm(packed2)

        # do the unpacking to batch size x max_length x embedding_dim
        unpacked1, _ = pad_packed_sequence(output1, batch_first=True)
        unpacked2, _ = pad_packed_sequence(output2, batch_first=True)
        # take maximum over sequence dim -> batch_size x embedding_dim 
        u = torch.max(unpacked1, dim=1)[0]
        v = torch.max(unpacked2, dim=1)[0]
        

        diff = torch.abs(u - v)
        dotprod = u * v
        combined = torch.hstack([u, v, diff, dotprod])
        logits = self.mlp(combined)
        return logits