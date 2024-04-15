import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess import *
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from models.bow import BOW
from models.lstm import LSTM_NLI
import datetime
# for logging
TIME = datetime.datetime.now()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.manual_seed(seed)

print(device)


        

labels = ['neutral', 'entailment', 'contradiction']
# map label to numeric
label_mapping = {label: index for index, label in enumerate(labels)}

train_split = preprocess(split='train')
vocab = create_vocab(train_split)
embeddings = align_vocab_with_glove(vocab)
#embeddings = np.random.rand(33623, 300)
#print(embeddings.shape)

# TextpairDataset and custom_collate function in preprocess.py
train_data = TextpairDataset(train_split, vocab, label_mapping)
print(f'Train set size: {len(train_data)}')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=custom_collate)
print(f'Number of train batches: {len(train_loader)}')

validation_split = preprocess(split='dev')
validation_data = TextpairDataset(validation_split, vocab, label_mapping)
print(f'Validation set size: {len(validation_data)}')

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=False, collate_fn=custom_collate)

embedding_dim = 300
vocab_size = len(vocab.mapping)
#vocab_size = 33623
print(f'size of vocab: {vocab_size}')
#hidden_dim = 512
# bow = BOW(embedding_dim, hidden_dim, vocab_size, len(labels) )
# bow.token_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
# bow.token_embeddings.weight.requires_grad = False
# bow.to(device)


def load_model(embeddings, labels, vocab_size, model_flag='lstm', state_file_path = None):
    embedding_dim = 300
    hidden_dim = 512
    if model_flag == 'lstm':
        # TODO: lstm_dim concatenation or not? 
        lstm_dim = 2048
        model = LSTM_NLI(embedding_dim, lstm_dim, hidden_dim, vocab_size, len(labels))
        model.token_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        model.token_embeddings.weight.requires_grad = False
        
    elif model_flag == 'bow':
        model = BOW(embedding_dim, hidden_dim, vocab_size, len(labels) )
        model.token_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        model.token_embeddings.weight.requires_grad = False

    if state_file_path:
        state_dict = torch.load(state_file_path)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    return model


# print("Model's state_dict:")
# sd = lstm.state_dict()
# for param_tensor in sd:
#     print(param_tensor, "\t", sd[param_tensor].size())

# del sd['token_embeddings.weight']
# print('deleted')
# print("Model's state_dict:")
# for param_tensor in sd:
#     print(param_tensor, "\t", sd[param_tensor].size())

# torch.save(sd, 'lstm_wo_embeddings.pth')

loss_module = nn.CrossEntropyLoss()

def evaluate(model, dataloader, loss_module, model_flag):
    model.eval()
    val_loss = 0
    total_preds, correct_preds = 0, 0
    with torch.no_grad():
        for i, (sent1s, sent2s, labels, lengths1, lengths2) in enumerate(dataloader, 0):
            sent1s = sent1s.to(device)
            sent2s = sent2s.to(device)
            labels = labels.to(device)
            if model_flag == 'bow':
                output = model(sent1s,sent2s)

            elif model_flag == 'lstm':
                output = model(sent1s,lengths1, sent2s, lengths2)
            predicted_labels = torch.argmax(output, dim=1)
            correct_preds += (predicted_labels == labels).sum() 
            total_preds += labels.shape[0]
            output_val_loss = loss_module(output, labels)
            val_loss += output_val_loss.item()
    return val_loss / len(dataloader), correct_preds / total_preds
    

def train(train_loader, validation_loader, model, loss_module, num_epochs=50, model_flag='bow'):
    writer = SummaryWriter(f'runs/{model_flag}/{TIME}')

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # learning rate * 0.99 after every epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    last_validation_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        print(f'starting epoch {epoch} with learning rate: {scheduler.get_last_lr()[0]}')
        for i, (sent1s, sent2s, labels, lengths1, lengths2) in enumerate(train_loader, 0):
            sent1s = sent1s.to(device)
            sent2s = sent2s.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if model_flag == 'bow':
                output = model(sent1s,sent2s)

            elif model_flag == 'lstm':
                output = model(sent1s,lengths1, sent2s, lengths2)
            # print('softmax output', output)
            # print('softmax output shape', output.shape)
            # print('labels', labels)
            # print('labels shape', labels.shape)
            output_loss = loss_module(output, labels)
            train_loss += output_loss.item()
            output_loss.backward()
            optimizer.step()

            # adapted from: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
            if i % 250 == 249:    # every 250 mini-batches
                writer.add_scalar('training loss',
                                train_loss / 250,
                                epoch * len(train_loader) + i)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_module, model_flag)
        
        print(f'avg training loss at epoch {epoch}: {validation_loss}')
        print(f'Validation accuracy at epoch {epoch}: {validation_accuracy }%')
        writer.add_scalar("Loss/validation", validation_loss, epoch)
        writer.add_scalar("Accuracy/validation", validation_accuracy, epoch)
        if validation_accuracy < last_validation_accuracy:
            print('accuracy decreased')
            for group in optimizer.param_groups:
                # decrease lr by 5 if validation accuracy decreases
                group['lr'] /= 5
        last_validation_accuracy = validation_accuracy
        scheduler.step()
        if scheduler.get_last_lr()[0] < 0.00001:
            print('reached treshold, stopping training')
            writer.flush()
            writer.close()
            sd = model.state_dict()
            # for param_tensor in sd:
            #     print(param_tensor, "\t", sd[param_tensor].size())

            del sd['token_embeddings.weight']
            torch.save(sd, f'{model_flag}_wo_embeddings.pth')

            break


model = load_model(embeddings, labels, vocab_size, 'lstm', 'lstm_wo_embeddings.pth')
#train(train_loader, validation_loader, lstm, loss_module, model_flag='lstm')