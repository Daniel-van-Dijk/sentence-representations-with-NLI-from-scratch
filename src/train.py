import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess import *
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.manual_seed(seed)

print(device)

v_glove, vectors = get_glove()

labels = ['neutral', 'entailment', 'contradiction']
label_mapping = {index: label for label, index in enumerate(labels)}



def prep(pair, vocab, label_mapping):
  x1 = []
  for token in pair['sentence_1']:
    x1.append(vocab.mapping.get(token.text, 0))
  x1 = torch.LongTensor([x1])
  x1 = x1.to(device)

  x2 = []
  for token in pair['sentence_2']:
    x2.append(vocab.mapping.get(token.text, 0))
  x2 = torch.LongTensor([x2])
  x2 = x2.to(device)

  y = torch.LongTensor([label_mapping[pair['gold_label']]])
  y = y.to(device)

  return x1, x2, y

dev_split = preprocess(split='dev')

print(dev_split[0])
pair = dev_split[0]
x1, x2, y = prep(pair, v_glove, label_mapping)
print(x1)
print(x2)
print(y)
#print(bow(x1))
# optimizer = optim.Adam(bow.parameters(), lr=0.0005)

