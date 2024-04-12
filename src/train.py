import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess import *
from torch.nn.utils.rnn import pad_sequence
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.manual_seed(seed)

print(device)

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
        

labels = ['neutral', 'entailment', 'contradiction']
# map label to numeric
label_mapping = {label: index for index, label in enumerate(labels)}


# For testing purposes, use dev split as training set:
# TODO: change dev to train

train_split = preprocess(split='dev')
vocab = create_vocab(train_split)
embeddings = align_vocab_with_glove(vocab)

# TextpairDataset and custom_collate function in preprocess.py
train_data = TextpairDataset(train_split, vocab, label_mapping)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=custom_collate)

validation_split = preprocess(split='test')
validation_data = TextpairDataset(validation_split, vocab, label_mapping)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=False, collate_fn=custom_collate)


embedding_dim = 300
vocab_size = len(vocab.mapping)
print(vocab_size)
hidden_dim = 512
bow = BOW(embedding_dim, hidden_dim, vocab_size, len(labels) )
bow.token_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
bow.token_embeddings.weight.requires_grad = False
bow.to(device)

loss_module = nn.CrossEntropyLoss()

def train(train_loader, validation_loader, model, loss_module, num_epochs=5):
    # TODO: add weight decay etc..
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for sent1s, sent2s, labels in train_loader:
            sent1s = sent1s.to(device)
            sent2s = sent2s.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = bow(sent1s, sent2s)
            # print('softmax output', output)
            # print('softmax output shape', output.shape)
            # print('labels', labels)
            # print('labels shape', labels.shape)
            output_loss = loss_module(output, labels)
            train_loss += output_loss
            output_loss.backward()
            optimizer.step()
        print(f'avg loss at epoch {epoch}: {train_loss / len(train_loader)}')
            

train(train_loader, validation_loader, bow, loss_module)