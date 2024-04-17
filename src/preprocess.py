import json
import pickle
import time
import os
from spacy.lang.en import English
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from utils import save_file, load_file

def read_json(split='train'):
    data = []
    with open(f"../data/snli_1.0/snli_1.0_{split}.jsonl", 'r') as json_file:
        for line in json_file:
                pair = json.loads(line)
                # only keep label and sentence pair
                data.append({key: pair[key] for key in ['gold_label', 'sentence1', 'sentence2']})
    print(f'done reading {split} json')
    return data

def preprocess(split='train'):
    """lowering and tokenization of sentences"""
    nlp = English()
    tokenizer = nlp.tokenizer
    data = read_json(split)
    preprocessed = []
    labels = ['neutral', 'entailment', 'contradiction']
    for pair in data:
        if pair['gold_label'] in labels:
            sent1 = list(tokenizer(pair['sentence1'].lower()))
            sent2 = list(tokenizer(pair['sentence2'].lower()))
            preprocessed.append({'sentence_1' : sent1, 
                                'sentence_2' : sent2, 
                                'gold_label' : pair['gold_label']})
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
    self.mapping = {token: index for index, token in enumerate(vocab_list)}

def create_vocab(dataset):
  v = Vocabulary()
  for pair in dataset:
    for token in pair['sentence_1'] + pair['sentence_2']:
      v.add_token(token.text)
  v.create_mapping()
  return v


def align_vocab_with_glove(data_vocab, embeddings_file='saved_files/glove_NLI_embeddings.npy'):
    glove_dim = 300
    if os.path.exists(embeddings_file):
      embeddings = np.load(embeddings_file)
      print("loaded embeddings from file")
      return embeddings
    # create zero's embedding matrix of size (vocab_size, glove_dim) 
    # vocab tokens not in glove will keep zero embeddings
    embeddings = np.zeros((len(data_vocab.mapping), glove_dim))
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
    np.save(embeddings_file, embeddings)
    print("saved embeddings")
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
          # token.text since token is still object from spacy
          # convert tokens to their id's and use 0 (<unk> index) if token not in vocab
          sent1.append(self.vocab.mapping.get(token.text, 0))

        sent2 = []
        for token in self.dataset[idx]['sentence_2']:
          sent2.append(self.vocab.mapping.get(token.text, 0))

        # map label to numeric
        label = self.label_mapping[self.dataset[idx]['gold_label']]
        
        return torch.LongTensor([sent1]), torch.LongTensor([sent2]), torch.LongTensor([label])
    


def custom_collate(batch):
    """ map all sentences to the maximum length within a batch"""
    # max_length across both sentences
    max_length = max([max(len(sent1[0]), len(sent2[0])) for sent1, sent2, _ in batch])
    sent1s, lengths1, sent2s, lengths2, labels = [], [], [], [], []
    # ID = 1 for pad token in vocab
    padding_ID = 1
    for sent1, sent2, label in batch:
      # fill difference between max length (of batch) and sentence length with padding tokens
      padded_sent1 = sent1.tolist()[0] + [padding_ID] * (max_length - len(sent1[0]))
      padded_sent2 = sent2.tolist()[0] + [padding_ID] * (max_length - len(sent2[0]))
      # store lengths of sentences for pack_padded_sequence
      lengths1.append(len(sent1[0])), lengths2.append(len(sent2[0]))
      sent1s.append(padded_sent1), sent2s.append(padded_sent2), labels.append(label)
    return torch.tensor(sent1s), torch.tensor(sent2s), torch.tensor(labels), lengths1, lengths2