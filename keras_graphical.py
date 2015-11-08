# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os,re
from keras.datasets import mnist
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
np.random.seed(1337)  # for reproducibility

#############
# LOAD DATA #
#############
def filesinroot(dir,wildcard,recursion):
    matchs=[]
    exts=wildcard.split()
    for root,subdirs,files in os.walk(dir):
        for name in files:
            for ext in exts:
                if(re.search(ext,name)):
                    matchs.append(name)
                    break
        if(not recursion):
            break
    return matchs
parent_path = os.path.split(os.path.realpath(__file__))[0]
path=parent_path+'/DataCut/'
matchs=filesinroot(path,"track",0)
parent_path = os.path.split(os.path.realpath(__file__))[0]
path=parent_path+'/DataCut/'
matchs=filesinroot(path,"track",0)
data=[]
for entry in matchs:
    data.append(np.loadtxt(path+entry,delimiter=','))
    
length=1
expcnt=0
for entry in data:
    expcnt=expcnt+entry.shape[0]/length
dataset=np.zeros((expcnt,length,8))  
cnt=0 
for entry in data:
    for i in range(entry.shape[0]/length):
        dataset[cnt,:,:]=entry[i*length:(i+1)*length,:]
        cnt=cnt+1
data=dataset
data=np.asarray(data,dtype='float32')
#shuffle
np.random.shuffle(data)
#scale
paramin=np.amin(np.amin(data[:,:,5:],axis=0),axis=0)[None,None,:]
paramax=np.amax(np.amax(data[:,:,5:],axis=0),axis=0)[None,None,:]
data[:,:,5:]=(data[:,:,5:]-paramin)/(paramax-paramin)
#categorize
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i,j,6]>0.9:
            data[i,j,6]=2
        elif data[i,j,6]>0.1 and data[i,j,6]<0.9:
            data[i,j,6]=1
        else:
            data[i,j,6]=0


###############
# BUILD MODEL #
###############

'''
    Train a simple deep NN on the MNIST dataset.

    Get to 98.30% test accuracy after 20 epochs (there is *a lot* of margin for parameter tuning).
    2 seconds per epoch on a GRID K520 GPU.
'''

batch_size = 40
nb_classes = 3
nb_epoch = 20

# the data, shuffled and split between tran and test sets
train_data,test_data=np.split(data,[int(0.8*data.shape[0])],axis=0)
X_train,y_train=np.split(train_data,[5],axis=2)
X_test, y_test=np.split(test_data,[5],axis=2)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train[:,0,1], nb_classes)
Y_test = np_utils.to_categorical(y_test[:,0,1], nb_classes)
Y_train1=y_train[:,0,0]
Y_test1=y_test[:,0,0]

graph=Graph()
graph.add_input(name='input', input_shape=(5,))
graph.add_node(Dense(100), name='dense1', input='input')
graph.add_node(Dense(1), name='dense2', input='dense1')
graph.add_node(Dense(3), name='dense3', input='dense1')
graph.add_node(Activation('softmax'), name='softmax1',input='dense3')
graph.add_output(name='output1', input='dense2')
graph.add_output(name='output2', input='softmax1')

graph.compile('rmsprop', {'output1':'mse', 'output2':'categorical_crossentropy'})
historylog = graph.fit({'input':X_train, 'output1':Y_train1, 'output2':Y_train},
                    batch_size=batch_size, nb_epoch=nb_epoch,verbose=2,validation_split=0.2)
score=graph.evaluate({'input':X_train, 'output1':Y_train1, 'output2':Y_train}, verbose=1)

#########

'''rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
sgd=SGD(lr=0.1, momentum=0., decay=0., nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])'''

'''
记录：
一.超参数表：网络方面：单层神经元数，batch大小，神经元类型，scaling(似乎没影响)
学习方面：lr
1.网络（5，128），（128，128），（128，3）得到
loss: 0.1598 - acc: 0.9486 - val_loss: 0.1962 - val_acc: 0.9363
2.网络（5，128），（128，3）得到
loss: 0.1699 - acc: 0.9463 - val_loss: 0.1645 - val_acc: 0.9490
3.网络（5，4），（4，3）得到
acc: 0.9469 - val_loss: 0.1720 - val_acc: 0.9495
4.网络（5，100），（100，3）得到，batch40,relu，无scale
loss: 0.1593 - acc: 0.9501 - val_loss: 0.1589 - val_acc: 0.9513
'''
'''
这怎么就会冲不到100%呢？！
添加对比，我真的只把速度和压力那两个有关的维度输入，会怎么样
'''
'''
work：
1.graphical model
2.data augumentation
'''