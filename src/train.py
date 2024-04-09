import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess import *
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

with open("/home/scur0220/assignment1/sentence-representations-with-NLI-from-scratch/src/SentEval/pretrained/glove.840B.300d.txt", 'r') as file:
    glove = file.readlines()

def get_glove(glove):
    v_glove = Vocabulary()
    vectors = []
    # <unk> and <pad> are zero-initialized
    # embeddings for <unk> and <pad>,
    vectors.append([0]*300)
    vectors.append([0]*300)
    for line in glove:
        token = line.split()[0]
        v_glove.count_token(token)
        embedding = line.split()[1:]
        vectors.append(embedding)
    v_glove.build()
    print("Vocabulary size:", len(v_glove.w2i))
    vectors = np.stack(vectors, axis=0).astype(float)
    return v_glove, vectors

v_glove, vectors = get_glove(glove[:10])


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

i2t = ['neutral', ]
t2i = OrderedDict({p : i for p, i in zip(i2t, range(len(i2t)))})
print(t2i)
print(t2i['very positive'])


# bow = BOW(len(v_glove.w2i), 300, 100, len(t2i), vocab=v_glove)

# # copy pre-trained word vectors into embeddings table
# bow.embed.weight.data.copy_(torch.from_numpy(vectors))
# bow.embed.weight.requires_grad = False
# pt_deep_cbow_model = bow.to(device)
# optimizer = optim.Adam(pt_deep_cbow_model.parameters(), lr=0.0005)

