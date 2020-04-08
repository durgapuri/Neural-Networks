#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import keras.utils
import tensorflow as tf
from mnist import MNIST
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import idx2numpy
import sys

class CNN:
    mnist_data = None
    md = None
    
    def fit_cnn_model(self,train_input, train_labels):
        self.md.add(Conv2D(28, kernel_size=(3,3), input_shape =(28, 28, 1)))
        self.md.add(MaxPooling2D(pool_size=(2,2)))
        self.md.add(Flatten())
        self.md.add(Dense(128, activation="relu"))
        self.md.add(Dropout(0.2))
        self.md.add(Dense(10, activation=tf.nn.softmax))
        self.md.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.md.fit(train_input, train_labels, epochs=20, batch_size=500, verbose=0)
        
    def train(self,train_path):
        self.mnist_data = MNIST(train_path)
        self.md = Sequential()
        train_input, train_labels = self.mnist_data.load_training()
        train_input = np.array(train_input)
        train_input = train_input.reshape(train_input.shape[0],28,28,1)
        train_labels = np.array(train_labels)
        train_input = train_input.astype('float32')
        train_labels = keras.utils.to_categorical(train_labels, 10)
        self.fit_cnn_model(train_input, train_labels)
        
    def predict(self,train_path):
        test_file_name = train_path+'/t10k-images-idx3-ubyte'
        test_input = idx2numpy.convert_from_file(test_file_name)
        test_input = test_input.reshape(test_input.shape[0],28,28,1)
        test_input = test_input.astype('float32')
        predictions = self.md.predict(test_input)
        predicted_labels= []

        for i in range(len(predictions)):
            predicted_labels.append(np.argmax(np.round(predictions[i])))
        return predicted_labels


mn = CNN()
# train_path = '/home/jyoti/Documents/SMAI/assign3/Q3'
train_path = sys.argv[1]
mn.train(train_path)
predicted_labels_cnn = mn.predict(train_path)
for val in predicted_labels_cnn:
    print(val)


# In[ ]:





# In[ ]:




