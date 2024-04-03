import json
import spacy
nlp = spacy.load("en_core_web_sm")

def read_json(split='dev'):
    data = []
    with open(f"../data/snli_1.0_{split}.jsonl", 'r') as json_file:
        for line in json_file:
            pair = json.loads(line)
            # only keep label and sentence pair
            data.append({key: pair[key] for key in ['gold_label', 'sentence1', 'sentence2']})
    return data

def preprocess(split='dev'):
    """lowering and tokenization of sentences"""
    data = read_json(split)
    for pair in data[:5]:
        pair['sentence1_t'] = list(nlp(pair['sentence1'].lower()))
        pair['sentence2_t'] = list(nlp(pair['sentence2'].lower()))
    print(data[0])
    return data
preprocess()
