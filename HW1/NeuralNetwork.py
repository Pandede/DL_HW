# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:23:34 2019

@author: Chan Chak Tong
"""

import numpy as np
import pandas as pd
#%%
# =============================================================================
# Question 1~2 - Neuron network and backpropagation
# =============================================================================
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.weight = []
        self.bias = []
    
    def __get_err_rate(self, y_true, y_pred):
        '''
        計算分類器錯誤率
        
        y_true: 真實資料的目標
        y_pred: 模型預測的目標
        
        回傳: 錯誤率
        '''
        return np.mean(~(np.argmax(y_true, axis=0) == np.argmax(y_pred, axis=0)))
    
    def add(self, layer):
        '''
        堆疊模型的神經層
        '''
        assert type(layer) == Layer, 'Type of layer is invalid.'
        self.layers.append(layer)
        
    def build(self, learning_rate=.1):
        '''
        初始化模型的權重W、偏差B及學習率LR
        '''
        self.learning_rate = learning_rate
        
        for i in range(len(self.layers)-1):
            link_size = (self.layers[i]._nodes, self.layers[i+1]._nodes)
            self.weight.append(np.random.normal(scale=np.sqrt(2 / np.sum(link_size)), size=link_size))
            self.bias.append(np.zeros((link_size[1], 1)))
            
    def train(self, X, Y, epochs=10, batch_size=1, validation_data=None):
        '''
        根據資料訓練模型
        
        X: 資料特徵
        Y: 目標
        epochs: 迭代次數
        batch_size: 每個Mini-batch的大小
        validation_data: 測試資料
        
        回傳: 訓練過程中的Training Loss, Training Acc, Testing loss, Testing acc
        '''
        assert X.shape[0] == self.layers[0]._nodes, 'The dimension of features is invalid.'
        assert Y.shape[0] == self.layers[-1]._nodes, 'The dimension of targets is invalid.'
        
        train_loss = np.zeros(epochs)
        train_err_rate = np.zeros(epochs)
        test_loss = np.zeros(epochs)
        test_err_rate = np.zeros(epochs)
        
        idx = np.arange(X.shape[1])
        for e in range(epochs):
            np.random.shuffle(idx)
            X, Y = X[:, idx], Y[:, idx]
            for b in range(0, X.shape[1], batch_size):
                x = X[:, b:b+batch_size]
                y = Y[:, b:b+batch_size]

                # Forward
                x_list = []
                z_list = [x]
                for i in range(len(self.weight)):
                    x = np.dot(self.weight[i].T, z_list[-1]) + self.bias[i]
                    x_list.append(x)
                    z = self.layers[i+1]._activation.function(x)
                    z_list.append(z)
                
                # Backward
                delta_w = [np.zeros_like(w) for w in self.weight]
                delta_b = [np.zeros_like(b) for b in self.bias]

                delta = Loss.prime(y, z) * self.layers[-1]._activation.prime(x)
                delta_b[-1] = np.mean(delta, axis=1, keepdims=True)
                delta_w[-1] = delta.dot(z_list[-2].T).T

                for i in range(2, len(self.layers)):
                    prime = self.layers[-i]._activation.prime(x_list[-i])
                    delta = self.weight[-i+1].dot(delta) * prime
                    delta_b[-i] = np.mean(delta, axis=1, keepdims=True)
                    delta_w[-i] = delta.dot(z_list[-i-1].T).T
                    
                for i in range(len(self.weight)):
                    self.weight[i] -= self.learning_rate * delta_w[i]
                    self.bias[i] -= self.learning_rate * delta_b[i]
                
            train_loss[e], train_err_rate[e] = self.evaluate(X, Y)
            
            if validation_data is not None:
                test_loss[e], test_err_rate[e] = self.evaluate(validation_data[0], validation_data[1])
            print('[Epoch %d  Train loss: %f Train err: %.2f%% Testing loss: %f Test err: %.2f%%]' % (e, train_loss[e], 100*train_err_rate[e], test_loss[e], 100*test_err_rate[e]))
            
        return (train_loss, train_err_rate) if validation_data is None else (train_loss, train_err_rate, test_loss, test_err_rate)
    
    def predict(self, X):
        '''
        模型預測
        回傳: 預測目標
        '''
        z = X
        for i in range(len(self.weight)):
            z = np.dot(self.weight[i].T, z) + self.bias[i]
            z = self.layers[i+1]._activation.function(z)
        
        return z

    def evaluate(self, X, Y):
        '''
        評估模型的準確性
        
        回傳: Loss, Accuracy
        '''
        z = self.predict(X)
        return Loss.function(Y, z), self.__get_err_rate(Y, z)
    
class Loss:
    '''
    預設Loss為Cross entropy
    '''
    @staticmethod
    def function(true_y, pred_y):
        return -np.sum(true_y * np.log(pred_y)) / true_y.shape[1]
    
    @staticmethod
    def prime(true_y, pred_y):
        return (pred_y - true_y) / true_y.shape[1]
    
class Layer:
    def __init__(self, nodes, activation='linear'):
        self._nodes = nodes
        self._activation = Activation(activation)
        
class Activation:
    '''
    激活函數
    '''
    def __new__(self, activation):
        if activation.capitalize() in self.__dict__:
            return eval('self.%s' % activation.capitalize())
        else:
            raise ValueError('The value of activation is invalid.')
    
    class Sigmoid:
        @staticmethod
        def function(X):
            return 1 / (1 + np.exp(-X))
        
        @classmethod
        def prime(self, X):
            return self.function(X) * (1 - self.function(X))
       
    class Linear:
        @staticmethod
        def function(X):
            return X
        
        @staticmethod
        def prime(X):
            return np.ones_like(X)
    
    class Relu:
        @staticmethod
        def function(X):
            return np.clip(X, 0, np.inf)
        
        @staticmethod
        def prime(X):
            return (X > 0).astype(np.int8)
    
    class Softmax:
        @staticmethod
        def function(X):
            return np.exp(X) / np.sum(np.exp(X), axis=0)
        
        @staticmethod
        def prime(X):
            return Activation.Sigmoid.prime(X)
        
    class Tanh:
        @staticmethod
        def function(X):
            return np.tanh(X)
        
        @classmethod
        def prime(self, X):
            return 1 - np.tanh(X) ** 2
    
#%%
data = pd.read_csv('./titanic.csv')
X = data.iloc[:, 1:].values.T
Y = pd.get_dummies(data.iloc[:, 0]).values.T

train_x, test_x = X[:, :800], X[:, 800:]
train_y, test_y = Y[:, :800], Y[:, 800:]

#%%
model1 = NeuralNetwork()
model1.add(Layer(6))
model1.add(Layer(3, activation='relu'))
model1.add(Layer(3, activation='relu'))
model1.add(Layer(2, activation='softmax'))
model1.build(learning_rate=0.01)
train_loss_1, train_err_rate_1, test_loss_1, test_err_rate_1 = \
model1.train(train_x, train_y, epochs=3000, batch_size=20, validation_data=[test_x, test_y])
#%%
import matplotlib.pyplot as plt
plt.plot(train_loss_1, label='Learning Curve')
plt.legend()
plt.show()
plt.plot(train_err_rate_1, label='Training error rate')
plt.plot(test_err_rate_1, label='Testing error rate')
plt.legend()
plt.show()
#%%
# =============================================================================
# Question 3 - Fare normalization
# =============================================================================
data['Fare'] = (data['Fare'] - data['Fare'].mean()) / data['Fare'].std()
data['Age'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()
#%%
X = data.iloc[:, 1:].values.T
Y = pd.get_dummies(data.iloc[:, 0]).values.T

train_x, test_x = X[:, :800], X[:, 800:]
train_y, test_y = Y[:, :800], Y[:, 800:]

model2 = NeuralNetwork()
model2.add(Layer(6))
model2.add(Layer(3, activation='sigmoid'))
model2.add(Layer(3, activation='sigmoid'))
model2.add(Layer(2, activation='softmax'))
model2.build(learning_rate=0.01)
train_loss_2, train_err_rate_2, test_loss_2, test_err_rate_2 = \
model2.train(train_x, train_y, epochs=1000, batch_size=20, validation_data=[test_x, test_y])
#%%
import matplotlib.pyplot as plt
plt.plot(train_loss_2, label='Learning Curve')
plt.legend()
plt.show()
plt.plot(train_err_rate_2, label='Training error rate')
plt.plot(test_err_rate_2, label='Testing error rate')
plt.legend()
plt.show()
#%%
# =============================================================================
# Question 4 - Most affective feature
# =============================================================================
# 數值越大代表對Y越大影響
corr = np.abs(data.corr().iloc[0])
#%%
# =============================================================================
# Question 5 - Categorical data
# =============================================================================
data = pd.get_dummies(data, columns=['Pclass'])

X = data.iloc[:, 1:].values.T
Y = pd.get_dummies(data.iloc[:, 0]).values.T

train_x, test_x = X[:, :800], X[:, 800:]
train_y, test_y = Y[:, :800], Y[:, 800:]

model3 = NeuralNetwork()
model3.add(Layer(8))
model3.add(Layer(30, activation='tanh'))
model3.add(Layer(30, activation='tanh'))
model3.add(Layer(2, activation='softmax'))
model3.build(learning_rate=0.01)
train_loss_3, train_err_rate_3, test_loss_3, test_err_rate_3 = \
model3.train(train_x, train_y, epochs=10000, batch_size=20, validation_data=[test_x, test_y])
#%%
import matplotlib.pyplot as plt
plt.plot(train_loss_3, label='Learning Curve')
plt.plot(test_loss_3, label='Testing curve')
plt.legend()
plt.show()
plt.plot(train_err_rate_3, label='Training error rate')
plt.plot(test_err_rate_3, label='Testing error rate')
plt.legend()
plt.show()
#%%
# =============================================================================
# Question 6 - Data augmentation
# =============================================================================
def makeup_pred(data):
    return model1.predict(np.array(list(data.values())).reshape((-1,1)))

# 會死亡的X
death_data = {
        'PClass': 3,
        'Sex': 1,
        'Age': 1,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 0
        }

# 會存活的X
survival_data = {
        'PClass': 3,
        'Sex': 0,
        'Age': 0,
        'SibSp': 2,
        'Parch': 0,
        'Fare': 200
        }

survival_data_pred = makeup_pred(survival_data)
death_data_pred = makeup_pred(death_data)
#%%
fig = plt.figure(figsize=(10, 10))
plt.matshow(data.corr(),fignum=1)
plt.colorbar()
plt.xticks(np.arange(7), list(data.columns))
plt.yticks(np.arange(7), list(data.columns))
plt.show()