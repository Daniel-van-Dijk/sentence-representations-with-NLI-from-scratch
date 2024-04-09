import json
import spacy
import time
from spacy.lang.en import English
import numpy as np

def read_json(split='dev'):
    data = []
    with open(f"../data/snli_1.0/snli_1.0_{split}.jsonl", 'r') as json_file:
        for line in json_file:
                pair = json.loads(line)
                # only keep label and sentence pair
                data.append({key: pair[key] for key in ['gold_label', 'sentence1', 'sentence2']})
    print('done reading json')
    return data

def preprocess(split='dev'):
    """lowering and tokenization of sentences"""

    nlp = English()
    tokenizer = nlp.tokenizer
    data = read_json(split)
    preprocessed = []

    # TODO: ask labels = ['neutral', 'entailment', 'contradiction', '-']
    labels = ['neutral', 'entailment', 'contradiction']
    start = time.time()
    for pair in data:
        if pair['gold_label'] in labels:
            preprocessed.append({'sentence_1' : list(tokenizer(pair['sentence1'].lower())), 
                                'sentence_2' : list(tokenizer(pair['sentence2'].lower())), 
                                'gold_label' : pair['gold_label']})
    end = time.time()
    print(f" It took {(end - start):.2f} seconds to tokenize {split} split")
    return preprocessed

class Vocabulary:
  """
  Create vocabulary by:
  - generating a set of the words in corpus
  - put in list with <unk> and <pad> and use index of this list as ID
  - sort vocab set such that ID's will be the same across runs.
  """
  def __init__(self):
    self.vocab = set()
    self.mapping = []

  def add_token(self, token):
    self.vocab.add(token)
  
  def create_mapping(self):
    # set <unk> and <pad> at position 0 and 1 and append sorted vocab set as list.
    vocab_list = ['<unk>', '<pad>'] + sorted(list(self.vocab))
    # make mapping where key is word and value is index in index list
    self.mapping = {index: word for word, index in enumerate(vocab_list)}


# v = Vocabulary()
# for pair in preprocessed:
#   for token in pair['sentence_1'] + pair['sentence_2']:
#     v.add_token(token.text)
# v.create_mapping()


def get_glove():
    with open("./SentEval/pretrained/glove.840B.300d.txt", 'r') as file:
      glove = file.readlines()
    v_glove = Vocabulary()
    vectors = []
    # <unk> and <pad> are zero-initialized
    vectors.append([0]*300) # <unk>
    vectors.append([0]*300) # <pad>
    for line in glove[:10000]:
        token = line.split()[0]
        v_glove.add_token(token)
        embedding = line.split()[1:]
        vectors.append(embedding)
    v_glove.create_mapping()
    print("Vocabulary size:", len(v_glove.mapping))
    vectors = np.stack(vectors, axis=0).astype(float)
    return v_glove, vectors