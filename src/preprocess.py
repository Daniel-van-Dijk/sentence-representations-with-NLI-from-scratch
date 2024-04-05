import json
import spacy
# execute: "$python -m spacy download en" if en_core_web_sm can not be found.
# see https://stackoverflow.com/questions/54334304/spacy-cant-find-model-en-core-web-sm-on-windows-10-and-python-3-5-3-anacon

# only include tokenization for speed
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "attribute_ruler"])

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
    # TODO: make spacy pipe or save to file
    for pair in data:
        preprocessed.append({'sentence_1' : list(nlp(pair['sentence1'].lower())), 
                             'sentence_2' : list(nlp(pair['sentence2'].lower())), 
                             'gold_label' : pair['gold_label']})
    return preprocessed
