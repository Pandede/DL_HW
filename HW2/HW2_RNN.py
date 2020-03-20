# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:19:39 2019

@author: Chan Chak Tong
"""

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
#%%
class Preprocessing:
    @staticmethod
    def Tokenize(sentences_df):
        return sentences_df.lower().split()
    
    @staticmethod
    def Dictionary(words_list):
        word_set = set()
        for sentence in np.squeeze(words_list.values):
            word_set.update(sentence)
        # Value = 0 is reserved for nothing.
        return {key: value+1 for value, key in enumerate(word_set)}

    @staticmethod
    def Vocabulary(token, dictionary, length=10):
        vocab = np.zeros((len(token), length))
        for idx, sentence in enumerate(np.squeeze(token.values)):
            sentence_len = len(sentence)
            index = list(map(lambda word: dictionary.setdefault(word, len(dictionary)+2), sentence))
            index.extend([0]*(length - sentence_len))
            vocab[idx] = index[:length]
        return vocab
    
length = 10
accepted = pd.read_excel('./ICLR_accepted.xlsx')
accepted.columns = ['Title']
accepted['Accept'] = 1
rejected = pd.read_excel('./ICLR_rejected.xlsx')
rejected.columns = ['Title']
rejected['Accept'] = 0

train_data = pd.concat((accepted[:550], rejected[:700]), axis=0)
test_data = pd.concat((accepted[550:], rejected[700:]), axis=0)

train_token = train_data['Title'].apply(Preprocessing.Tokenize)
test_token = test_data['Title'].apply(Preprocessing.Tokenize)

dictionary = Preprocessing.Dictionary(train_token)
train_x = Preprocessing.Vocabulary(train_token, dictionary)
test_x = Preprocessing.Vocabulary(test_token, dictionary)
train_y = train_data['Accept'].values
test_y = test_data['Accept'].values
#%%
class RNN(nn.Module):
    def __init__(self, embed_size=100, nodes=30, layers=3, types='RNN'):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(len(dictionary)+2, embed_size)
        if types == 'RNN':
            self.rnn = nn.RNN(embed_size, nodes, layers)
        elif types == 'LSTM':
            self.rnn = nn.LSTM(embed_size, nodes, layers)
        else:
            raise ValueError('The type of RNN is not available')
        self.fc1 = nn.Linear(nodes, 16)
        self.fc2 = nn.Linear(16, 8)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(8, 2)
        
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        
    def forward(self, x):
        h = self.embedding(x.t())
        h, _ = self.rnn(h)
        h = self.fc1(h[-1])
        h = self.tanh(h)
        h = self.fc2(h)
        h = self.tanh(h)
        return self.output(h)
    
    def accuracy(self, y_true, y_pred):
        y_true = y_true.numpy()
        y_pred = y_pred.detach().numpy()
        identity = np.sum(y_true == np.argmax(y_pred, axis=1))
        return identity / len(y_true)
    
    def fit(self, train_x, train_y, test_x, test_y, epochs=300, batch_size=64):
        history = np.zeros((epochs, 4))
        for e in range(epochs):
            random_idx = np.arange(len(train_x))
            np.random.shuffle(random_idx)
            train_x = train_x[random_idx]
            train_y = train_y[random_idx]
            
            mini_train_loss, mini_train_acc = [], []
            self.train()
            for batch_idx in range(0, len(train_x), batch_size):
                batch_slice = slice(batch_idx, batch_idx+batch_size)
                x, y = train_x[batch_slice], train_y[batch_slice]
                x = torch.from_numpy(x).type('torch.LongTensor')
                y = torch.from_numpy(y).type('torch.LongTensor')
                
                train_prediction = self(x)
                loss = self.loss_func(train_prediction, y)
                mini_train_loss.append(loss.data.tolist())
                mini_train_acc.append(self.accuracy(y, train_prediction))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            train_loss = np.mean(mini_train_loss)
            train_acc = np.mean(mini_train_acc)
            
            self.eval()
            x = torch.from_numpy(test_x).type('torch.LongTensor')
            y = torch.from_numpy(test_y).type('torch.LongTensor')
            
            test_prediction = self(x)
            test_loss = self.loss_func(test_prediction, y)
                
            test_loss = test_loss.data.tolist()
            test_acc = self.accuracy(y, test_prediction)
            
            print('%d [Train - Loss: %f, Acc: %f] [Test - Loss: %f, Acc: %f]' % (e, train_loss, train_acc, test_loss, test_acc))
            history[e] = np.array([train_loss, train_acc, test_loss, test_acc])
        return history
epochs = 300
batch_size = 32

rnn_classifier = RNN()
history_rnn = rnn_classifier.fit(train_x, train_y, test_x, test_y)

lstm_classifier = RNN(types='LSTM')
history_lstm = lstm_classifier.fit(train_x, train_y, test_x, test_y)
#%%
import matplotlib.pyplot as plt
plt.plot(history_rnn.T[0], 'r', label='Train')
plt.title('RNN Learning Curve')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history_rnn.T[1], 'r', label='Train')
plt.plot(history_rnn.T[3], 'b', label='Test')
plt.title('RNN Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history_lstm.T[0], 'r', label='Train')
plt.title('LSTM Learning Curve')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


plt.plot(history_lstm.T[1], 'r', label='Train')
plt.plot(history_lstm.T[3], 'b', label='Test')
plt.title('LSTM Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()