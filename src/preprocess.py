import json
import spacy
import time
from spacy.lang.en import English
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn as nn

def read_json(split='dev'):
    data = []
    with open(f"../data/snli_1.0/snli_1.0_{split}.jsonl", 'r') as json_file:
        for line in json_file:
                pair = json.loads(line)
                # only keep label and sentence pair
                data.append({key: pair[key] for key in ['gold_label', 'sentence1', 'sentence2']})
    print(f'done reading {split} json')
    return data

def preprocess(split='dev'):
    """lowering and tokenization of sentences"""

    nlp = English()
    tokenizer = nlp.tokenizer
    data = read_json(split)
    preprocessed = []

    # TODO: ask "-"? labels = ['neutral', 'entailment', 'contradiction', '-']
    labels = ['neutral', 'entailment', 'contradiction']
    start = time.time()
  
    for pair in data:
        if pair['gold_label'] in labels:
            sent1 = list(tokenizer(pair['sentence1'].lower()))
            sent2 = list(tokenizer(pair['sentence2'].lower()))
            preprocessed.append({'sentence_1' : sent1, 
                                'sentence_2' : sent2, 
                                'gold_label' : pair['gold_label']})
    end = time.time()
    print(f" It took {(end - start):.2f} seconds to tokenize {split} split")
    return preprocessed

class Vocabulary:
  """
  Create vocabulary by:
  - generating a set of the words in corpus
  - sort vocab set such that ID's will be the same across runs.
  - put in list with <unk> and <pad> as first tokens and use index of this list as ID
  """
  def __init__(self):
    self.vocab = set()
    self.mapping = []

  def add_token(self, token):
    self.vocab.add(token)
  
  def create_mapping(self):
    # set <unk> and <pad> at position 0 and 1 and append sorted vocab set as list.
    vocab_list = ['<unk>', '<pad>'] + sorted(list(self.vocab))
    # make mapping where key is token and value is index in index list
    self.mapping = {index: token for token, index in enumerate(vocab_list)}


# v = Vocabulary()
# preprocessed = preprocess()
# for pair in preprocessed:
#   for token in pair['sentence_1'] + pair['sentence_2']:
#     v.add_token(token.text)
# v.create_mapping()


def align_vocab_with_glove(data_vocab):
    glove_dim = 300
    # create zero's embedding matrix of size (vocab_size, glove_dim) 
    # tokens from vocab which are not in glove will keep zero embeddings
    embeddings = np.zeros((len(data_vocab.mapping), glove_dim))
    start = time.time()
    with open("./SentEval/pretrained/glove.840B.300d.txt", 'r') as glove:
      for line in glove:
        # token is first element of line
        token = line.split()[0]
        # rest of line is 300-dim embedding
        glove_embedding = line.split()[1:]
        # only store embedding when token in vocab and embedding is correct dim
        if token in data_vocab.mapping and len(glove_embedding) == glove_dim:
            token_ID = data_vocab.mapping[token]
            # use token ID as row index in embedding matrix, convert string to float
            embeddings[token_ID, :] = np.array(glove_embedding, dtype=np.float32)
    end = time.time()
    print(f" It took {(end - start):.2f} seconds to align vocab with glove")
    return embeddings

# print(align_vocab_with_glove(v))


class TextpairDataset(Dataset):
    def __init__(self, dataset, vocab, label_mapping):
        self.dataset = dataset
        self.vocab = vocab
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sent1 = []
        for token in self.dataset[idx]['sentence_1']:
          sent1.append(self.vocab.mapping.get(token.text, 0))

        sent2 = []
        for token in self.dataset[idx]['sentence_2']:
          sent2.append(self.vocab.mapping.get(token.text, 0))

        label = self.label_mapping[self.dataset[idx]['gold_label']]
        
        return torch.LongTensor([sent1]), torch.LongTensor([sent2]), torch.LongTensor([label])
    


def custom_collate(batch):
    """ map all sentences to the maximum length within a batch"""
    # max_length across both sentences
    max_length = max([max(len(sent1[0]), len(sent2[0])) for sent1, sent2, _ in batch])
    sent1s = []
    sent2s = []
    labels = []
    # ID = 1 for pad token in vocab
    padding_ID = 1
    for sent1, sent2, label in batch:
      # fill difference between max length and sentence length with padding token
      padded_sent1 = sent1.tolist()[0] + [padding_ID] * (max_length - len(sent1[0]))
      padded_sent2 = sent2.tolist()[0] + [padding_ID] * (max_length - len(sent2[0]))
      sent1s.append(padded_sent1), sent2s.append(padded_sent2), labels.append(label)
    return torch.tensor(sent1s), torch.tensor(sent2s), torch.tensor(labels)