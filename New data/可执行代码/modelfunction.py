# -*- coding: utf-8 -*-
'''
build model funtion,and train function 
author: CUI HAOTIAN
date:2015.12.05
'''
import numpy as np
import cPickle,gzip,os 
from datetime import datetime
from matplotlib import pyplot as plt
from utils import loadgz,add_simu_data
today=datetime.today()
tstr=today.strftime('%y%m%d')

#from keras.datasets import mnist
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.initializations import uniform
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint

def build_mlp(in_dim, out_dim, h0_dim, h1_dim=None):
    model = Sequential() 
    model.add(Dense(h0_dim, activation='relu',input_dim=in_dim))  
    if h1_dim is not None:
        model.add(Dense(h1_dim,activation='relu'))
    #model.add(Dense(out_dim,W_regularizer=l2(0.0005)))
    model.add(Dense(out_dim))  
    model.add(Activation("linear"))  
    model.compile(loss="mse", optimizer="rmsprop")  
    return model
    
def train(X_train, y_train, X_test, y_test, model, batch_size=128, nb_epoch=300):
    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    path=parent_path+"/MLP_weights"+tstr+".hdf5"
    checkpointer = ModelCheckpoint(filepath=path, verbose=1, save_best_only=True)
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              nb_epoch=nb_epoch, 
              show_accuracy=False, 
              verbose=2, 
              validation_data=(X_test, y_test), 
              callbacks=[early_stopping, checkpointer])

def build_mlp_dataset(data, test=None,pred_range=[2,42], valid_pct=1./8):
    if test==None:
        #np.random.shuffle(data)
        train_pct = 1. - valid_pct
        train_data = data[:data.shape[0]*train_pct]
        valid_data = data[data.shape[0]*train_pct:]
    else:
        train_data = data
        valid_data = test
        
    print "trainset.shape, testset.shape =", train_data.shape, valid_data.shape
    X_train, y_train = np.hsplit(train_data,[5])
    X_valid, y_valid = np.hsplit(valid_data,[5])
                                           
    X_train = normalize(X_train)
    X_valid = normalize(X_valid)
    print 'X_train.shape, y_train.shape =', X_train.shape, y_train.shape
    return X_train, y_train, X_valid, y_valid
    
def normalize(X_train):
    X_train = X_train-np.array([[0.,0.,1.9,0.,2.]])
    X_train = X_train/np.array([[7.7,10.0,1,0.03,0.5]])

    return X_train.astype('float32')
    
def test(array,model):
    array=normalize(array)
    return model.predict(array,array.shape[0])
    
def visual_test(model,sequence=None):
    parent_path=os.path.split(os.path.realpath(__file__))[0]
    if sequence==None:
        f = gzip.open(parent_path+'/dataset/sequence.pkl.gz', 'rb')
        sequence=cPickle.load(f)
        f.close()
    targets=sequence[:,5]
    predictions=test(sequence[:,:5],model)
    plt.figure('test')
    plt.plot(targets,label='turn angle out')
    #plt.plot(3.*sequence[:,0],label='lean angle')
    #plt.plot(3.*sequence[:,1],label='lean angle rate')
    plt.plot(predictions,label='offline network out')
    plt.grid()
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.xlabel('time(10ms)')
    plt.ylabel('Value')
    plt.show()
    print 'abs error',np.mean(np.abs(predictions-targets))
        
    
def convert_norm(X):
    return X*np.array([[7.7,10.0,1,0.03,0.5]])+np.array([[0.,0.,1.9,0.,2.]])

if __name__=='__main__':
    parent_path = os.path.split(os.path.realpath(__file__))[0]
    trainset=loadgz(parent_path+'/dataset/trainset.pkl.gz')
    testset=loadgz(parent_path+'/dataset/testset.pkl.gz')
    trainset=add_simu_data(trainset,trainset.shape[0]/6)
    testset=add_simu_data(testset,testset.shape[0]/6)
    X_train, y_train, X_valid, y_valid = build_mlp_dataset(data=trainset,test=testset)
    MLPmodel=build_mlp(X_train.shape[-1] ,y_train.shape[-1], 100,20)
    #是否加载pretrain model
    #MLPmodel.load_weights(parent_path+'/MLP_weightsBest.hdf5')
    #print X_train[:1024],y_train[:1024],X_valid[:1024],y_valid[:1024]
    #print X_train[:1024].mean(axis=0)
    train(X_train, y_train, X_valid, y_valid, MLPmodel, batch_size=128)
    #下面这个函数里可以选择可视化哪个数据文件
    visual_test(MLPmodel,sequence=testset)
    

'''只是改变了训练数据的比例，就得到了一组更好的模型'''    
'''[[-1.1058085   0.0181      0.          0.          1.89999998]] :0.342671066523

[[ -3.60963404e-01   1.79999997e-03   0.00000000e+00   0.00000000e+00
    1.89999998e+00]] :
    
[[ -8.02140906e-02   1.99999995e-04   1.89999998e+00   0.00000000e+00
    0.00000000e+00]]'''


