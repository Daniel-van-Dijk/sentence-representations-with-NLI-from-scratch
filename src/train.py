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


labels = ['neutral', 'entailment', 'contradiction']
label_mapping = {index: label for label, index in enumerate(labels)}


dev_split = preprocess(split='dev')

v = Vocabulary()
for pair in dev_split:
  for token in pair['sentence_1'] + pair['sentence_2']:
    v.add_token(token.text)
v.create_mapping()

embeddings = align_vocab_with_glove(v)

# TextpairDataset and custom_collate function in preprocess.py
Textpair_data = TextpairDataset(dev_split, v, label_mapping)

validation_loader = torch.utils.data.DataLoader(Textpair_data, batch_size=2, shuffle=False, collate_fn = custom_collate)

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
        


embedding_dim = 300
vocab_size = len(v.mapping)
print(vocab_size)
hidden_dim = 512
bow = BOW(embedding_dim, hidden_dim, vocab_size, len(labels) )
#embeddings = np.random.rand(vocab_size, embedding_dim)
bow.token_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
bow.token_embeddings.weight.requires_grad = False

loss = nn.CrossEntropyLoss()

def train(train_loader, model):
    # TODO: add weight decay etc..
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for sent1s, sent2s, labels in train_loader:
        optimizer.zero_grad()
        output = bow(sent1s, sent2s)
        # print('softmax output', output)
        # print('softmax output shape', output.shape)
        # print('labels', labels)
        # print('labels shape', labels.shape)
        output_loss = loss(output, labels)
        output_loss.backward()
        optimizer.step()
        print(output_loss)

train(validation_loader, bow)