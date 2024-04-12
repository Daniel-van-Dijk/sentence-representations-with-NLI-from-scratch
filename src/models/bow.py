import torch
import torch.nn as nn
import torch.nn.functional as F
class BOW(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_labels):
        super(BOW, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # dim * 4 as we concatenate 4 vectors: u, v, |u - v|, u*v
        self.linear1 = nn.Linear(embedding_dim*4, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_labels)
        self.relu = torch.nn.ReLU()

    def forward(self, sentence1, sentence2):
        # print(sentence1.shape)
        embeds1 = self.token_embeddings(sentence1)
        embeds2 = self.token_embeddings(sentence2)
        # take mean over sequence dim 
        u = embeds1.mean(1)
        v = embeds2.mean(1)
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