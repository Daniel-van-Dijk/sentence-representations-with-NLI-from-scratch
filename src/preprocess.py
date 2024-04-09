import json
import spacy
import time
from collections import Counter, OrderedDict, defaultdict
from spacy.lang.en import English
# # execute: "$python -m spacy download en" if en_core_web_sm can not be found.
# # see https://stackoverflow.com/questions/54334304/spacy-cant-find-model-en-core-web-sm-on-windows-10-and-python-3-5-3-anacon

# # only include tokenization for speed
# nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "attribute_ruler"])


nlp = English()
tokenizer = nlp.tokenizer

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

# Below is taken from assignment 2 of NLP1 

class OrderedCounter(Counter, OrderedDict):
  """Counter that remembers the order elements are first seen"""
  def __repr__(self):
    return '%s(%r)' % (self.__class__.__name__,
                      OrderedDict(self))
  def __reduce__(self):
    return self.__class__, (OrderedDict(self),)


class Vocabulary:
  """A vocabulary, assigns IDs to tokens"""

  def __init__(self):
    self.freqs = OrderedCounter()
    self.w2i = {}
    self.i2w = []

  def count_token(self, t):
    self.freqs[t] += 1

  def add_token(self, t):
    self.w2i[t] = len(self.w2i)
    self.i2w.append(t)

  def build(self, min_freq=0):
    '''
    min_freq: minimum number of occurrences for a word to be included
              in the vocabulary
    '''
    self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
    self.add_token("<pad>")  # reserve 1 for <pad> 

    tok_freq = list(self.freqs.items())
    tok_freq.sort(key=lambda x: x[1], reverse=True)
    for tok, freq in tok_freq:
      if freq >= min_freq:
        self.add_token(tok)
