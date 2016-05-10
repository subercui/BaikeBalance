# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import gzip,cPickle
f=gzip.open('dataset.pkl.gz','rb')
data=cPickle.load(f)
data=np.asarray(data,dtype='float32')
np.random.shuffle(data)
paramin=np.amin(np.amin(data,axis=0),axis=0)[None,None,:]
paramax=np.amax(np.amax(data,axis=0),axis=0)[None,None,:]
data=(data-paramin)/(paramax-paramin)

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.initializations import normal, identity
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils

batch_size = 30
nb_out = 3
nb_epochs = 600
hidden_units = 10

learning_rate = 0.001
clip_norm = 1.0
BPTT_truncate = 40

train_data,test_data=np.split(data,[int(0.8*data.shape[0])],axis=0)
X_train,Y_train=np.split(train_data,[5],axis=2)
X_test, Y_test=np.split(test_data,[5],axis=2)

#X_train = X_train.reshape(X_train.shape[0], -1, 1)
#X_test = X_test.reshape(X_test.shape[0], -1, 1)
#X_train = X_train.astype("float32")
#X_test = X_test.astype("float32")
#X_train /= 255
#X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


print('Evaluate IRNN...')
model = Sequential()
#activation 可以考虑
model.add(SimpleRNN(input_dim=5, output_dim=hidden_units,return_sequences=True,
                    init=lambda shape: normal(shape, scale=0.001),
                    inner_init=lambda shape: identity(shape, scale=1.0),
                    activation='relu', truncate_gradient=BPTT_truncate))
model.add(Dense(hidden_units, nb_out))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='mean_absolute_error', optimizer=rmsprop)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])
'''
print('Compare to LSTM...')
model = Sequential()
model.add(LSTM(1, hidden_units))
model.add(Dense(hidden_units, nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('LSTM test score:', scores[0])
print('LSTM test accuracy:', scores[1])'''
