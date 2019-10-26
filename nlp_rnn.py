#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:08:56 2019

@author: saurabhjain
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D
from keras.layers import LSTM
from sklearn.metrics.classification import classification_report
from sklearn.model_selection import train_test_split

max_features = 2000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 500
# Try different values of this to tune the output
embed_dim = 128
lstm_out = 196

# Importing the dataset
data = pd.read_csv('labeled_data_new - labeled_data.tsv', delimiter = '\t', quoting = 3)

# Data Summary
data.info()
data.label.value_counts()
fig = plt.figure(figsize = (6,4))
data.groupby('label').tweet.count().plot.bar(ylim=0)
plt.show()


# Data preprocessing
# Cleaning the texts
data['tweet'] = data['tweet'].apply(lambda x: str(x).lower())
data['tweet'] = data['tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

for idx,row in data.iterrows():
    row[1] = row[1].replace('rt',' ')


# Tokenize the text
tokenizer = Tokenizer(num_words = max_features, split=' ')
tokenizer.fit_on_texts(data['tweet'].values)
X = tokenizer.texts_to_sequences(data['tweet'].values)
X = pad_sequences(X, maxlen = 300)
Y = pd.get_dummies(data['label']).values

#Y = data.iloc[:,0:1].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

history = model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 1, validation_split=0.3)

y_pred = model.predict(X_test)
y_final = []
for i in range(len(y_pred)):
    y_final.append(np.argmax(y_pred[i]))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test[:,1], y_final)
evalRep =  classification_report(Y_test[:,1], y_final)


# Validation
validation_size = 1000

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


# Result Visualisation
history_dict = history.history
print(history_dict.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


twt = ['Bitch deserved it']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=300, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 1)[0]
if(np.argmax(sentiment) == 0):
    print("non offensive")
elif (np.argmax(sentiment) == 1):
    print("offensive")