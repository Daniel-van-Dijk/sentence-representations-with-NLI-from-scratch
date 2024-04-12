import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_labels):
        super(LSTM, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
        # dim * 4 as we concatenate 4 vectors: u, v, |u - v|, u*v
        self.linear1 = nn.Linear(embedding_dim*4, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_labels)
        self.relu = torch.nn.ReLU()

    def forward(self, sentence1, lengths1, sentence2, lengths2):
        # print(sentence1.shape)
        embeds1 = self.token_embeddings(sentence1)
        embeds2 = self.token_embeddings(sentence2)

        # pack padded variable input
        packed1 = pack_padded_sequence(embeds1, lengths1, batch_first=True, enforce_sorted=False)
        packed2 = pack_padded_sequence(embeds2, lengths2, batch_first=True, enforce_sorted=False)

        _, u, _ = self.lstm(packed1)
        _, v, _ = self.lstm(packed2)

        diff = torch.abs(u - v)
        dotprod = u * v
        # print(u.shape)
        # print(v.shape)
        # print(diff.shape)
        # print(dotprod.shape)
        combined = torch.hstack([u, v, diff, dotprod])
        intermediate = self.relu(self.linear1(combined))
        output = F.softmax(self.linear2(intermediate), dim=1)
        # print(combined.shape)
        # print(output)
        # print(output.shape)
        # print(combined.shape)
        return output