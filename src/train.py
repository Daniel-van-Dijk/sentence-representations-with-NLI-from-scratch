import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess import *
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def get_glove():
    with open("/home/scur0220/assignment1/sentence-representations-with-NLI-from-scratch/src/SentEval/pretrained/glove.840B.300d.txt", 'r') as file:
      glove = file.readlines()
    v_glove = Vocabulary()
    vectors = []
    # <unk> and <pad> are zero-initialized
    # embeddings for <unk> and <pad>,
    vectors.append([0]*300)
    vectors.append([0]*300)
    for line in glove[:100]:
        token = line.split()[0]
        v_glove.count_token(token)
        embedding = line.split()[1:]
        vectors.append(embedding)
    v_glove.build()
    print("Vocabulary size:", len(v_glove.w2i))
    vectors = np.stack(vectors, axis=0).astype(float)
    return v_glove, vectors

v_glove, vectors = get_glove()


class BOW(nn.Module):
  """Averaging pre-trained word embeddings"""

  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
    super(BOW, self).__init__()
    self.vocab = vocab
    self.embed = nn.Embedding(vocab_size, embedding_dim)
    self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)
    self.linear = nn.Linear(embedding_dim, output_dim)

  def forward(self, inputs):
    embeds = self.embed(inputs)
    logits = embeds.sum(1)  + self.bias

    logits = self.linear(logits)

    return logits
  
seed = 42
torch.manual_seed(seed)

i2t = labels = ['neutral', 'entailment', 'contradiction']
t2i = OrderedDict({p : i for p, i in zip(i2t, range(len(i2t)))})
print(t2i)
print(t2i['neutral'])


bow = BOW(len(v_glove.w2i), 300, 100, len(t2i), vocab=v_glove)

# copy pre-trained word vectors into embeddings table
bow.embed.weight.data.copy_(torch.from_numpy(vectors))
bow.embed.weight.requires_grad = False
bow = bow.to(device)


def prep(pair, vocab, t2i):
  # vocab returns 0 if the word is not there (i2w[0] = <unk>)
  x1 = [vocab.w2i.get(t, 0) for t in pair['sentence_1']]
  x1 = torch.LongTensor([x1])
  x1 = x1.to(device)

  x2 = [vocab.w2i.get(t, 0) for t in pair['sentence_2']]
  x2 = torch.LongTensor([x2])
  x2 = x2.to(device)

  y = torch.LongTensor([t2i[pair['gold_label']]])
  y = y.to(device)

  return x1, x2, y

dev_split = preprocess(split='dev')

print(dev_split[0])
pair = dev_split[0]
print(prep(pair, bow.vocab, t2i))
# optimizer = optim.Adam(bow.parameters(), lr=0.0005)

