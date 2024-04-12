import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess import *
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models.bow import BOW
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.manual_seed(seed)

print(device)


        

labels = ['neutral', 'entailment', 'contradiction']
# map label to numeric
label_mapping = {label: index for index, label in enumerate(labels)}

train_split = preprocess(split='dev')
vocab = create_vocab(train_split)
embeddings = align_vocab_with_glove(vocab)

# TextpairDataset and custom_collate function in preprocess.py
train_data = TextpairDataset(train_split, vocab, label_mapping)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=custom_collate)

validation_split = preprocess(split='test')
validation_data = TextpairDataset(validation_split, vocab, label_mapping)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=False, collate_fn=custom_collate)


embedding_dim = 300
vocab_size = len(vocab.mapping)
print(vocab_size)
hidden_dim = 512
bow = BOW(embedding_dim, hidden_dim, vocab_size, len(labels) )
bow.token_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
bow.token_embeddings.weight.requires_grad = False
bow.to(device)

loss_module = nn.CrossEntropyLoss()

def evaluate(model, dataloader, loss_module):
    model.eval()
    val_loss = 0
    total_preds, correct_preds = 0, 0
    with torch.no_grad():
        for sent1s, sent2s, labels, lengths1, lengths2 in dataloader:
            sent1s = sent1s.to(device)
            sent2s = sent2s.to(device)
            labels = labels.to(device)
            output = model(sent1s, sent2s)
            predicted_labels = torch.argmax(output, dim=1)
            correct_preds += (predicted_labels == labels).sum() 
            total_preds += labels.shape[0]
            output_val_loss = loss_module(output, labels)
            val_loss += output_val_loss.item()
    return val_loss / len(dataloader), correct_preds / total_preds
    

def train(train_loader, validation_loader, model, loss_module, num_epochs=1):
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # learning rate * 0.99 after every epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        print(f'starting epoch {epoch} with learning rate: {scheduler.get_last_lr()}')
        for sent1s, sent2s, labels, lengths1, lengths2 in train_loader:
            sent1s = sent1s.to(device)
            sent2s = sent2s.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(sent1s, sent2s)

            # print('softmax output', output)
            # print('softmax output shape', output.shape)
            # print('labels', labels)
            # print('labels shape', labels.shape)
            output_loss = loss_module(output, labels)
            train_loss += output_loss.item()
            output_loss.backward()
            optimizer.step()
        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_module)

        print(f'avg training loss at epoch {epoch}: {validation_loss}')
        print(f'Validation accuracy at epoch {epoch}: {validation_accuracy}')
       
        scheduler.step()
train(train_loader, validation_loader, bow, loss_module)