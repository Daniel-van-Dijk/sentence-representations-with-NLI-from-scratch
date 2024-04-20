import torch
from preprocess import *
from models.bow import BOW
from models.lstm import LSTM_NLI
from models.bilstm import BiLSTM_NLI
from models.bilstm_maxpool import BiLSTM_MaxPool_NLI
from models.lstm_maxpool import LSTM_MaxPool_NLI

def load_model(embeddings, labels, vocab_size, device, model_flag='lstm', state_file_path = None):
    embedding_dim = 300
    hidden_dim = 512
    print(f'loading {model_flag}')
    if model_flag == 'lstm':
        lstm_dim = 2048
        model = LSTM_NLI(embedding_dim, lstm_dim, hidden_dim, vocab_size, len(labels))
    elif model_flag == 'lstm_max':
        lstm_dim = 2048
        model = LSTM_MaxPool_NLI(embedding_dim, lstm_dim, hidden_dim, vocab_size, len(labels))
    elif model_flag == 'bilstm':
        bilstm_dim = 2048
        model = BiLSTM_NLI(embedding_dim, bilstm_dim, hidden_dim, vocab_size, len(labels))
    elif model_flag == 'bilstm_max':
        bilstm_dim = 2048
        model = BiLSTM_MaxPool_NLI(embedding_dim, bilstm_dim, hidden_dim, vocab_size, len(labels))
    elif model_flag == 'bow':
        model = BOW(embedding_dim, hidden_dim, vocab_size, len(labels) )
    model.token_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
    #keep embeddings fixed during training
    model.token_embeddings.weight.requires_grad = False
    if state_file_path:
        if str(device) == 'cpu':
            state_dict = torch.load(state_file_path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(state_file_path)
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    return model

def get_metrics(results):
    """ Computes micro and macro metrics of transfert tasks """
    micro, macro, num_tasks, num_samples = 0, 0, 0, 0
    for task in results:
        # only tasks that have accuracy as metric
        if 'devacc' in results[task]:
            macro += results[task]['devacc']
            # weight by number of samples
            micro += (results[task]['devacc'] * results[task]['ndev'])
            num_tasks += 1
            num_samples += results[task]['ndev']
    micro /= num_samples
    macro /= num_tasks
    return micro, macro


def save_file(file, filename):
    path = f'saved_files/{filename}'
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(file, f)

def load_file(filename):
    path = f'saved_files/{filename}'
    print(path)
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file
