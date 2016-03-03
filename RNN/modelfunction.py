# -*- coding: utf-8 -*-
#!!!!之后一定把stateful用起来，
#用stateful就要注意batch，可能batch等于1，
#还要注意什么时候断句并reset
'''
build model funtion,and train function 
author: CUI HAOTIAN
date:2015.12.05
'''
import numpy as np
import cPickle,gzip,os 
from theano import tensor as T
from datetime import datetime
from matplotlib import pyplot as plt
from makernnset import makernnset
from utils import loadgz
today=datetime.today()
tstr=today.strftime('%y%m%d')
del today

#from keras.datasets import mnist
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation
#import ipdb; ipdb.set_trace() # BREAKPOINT
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.initializations import uniform
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint

def load_weights(filepath):
    '''Load all layer weights from a HDF5 save file.
    '''
    import h5py
    f = h5py.File(filepath)
    weights=[]
    for k in range(f.attrs['nb_layers']):
        # This method does not make use of Sequential.set_weights()
        # for backwards compatibility.
        g = f['layer_{}'.format(k)]
        for p in range(g.attrs['nb_params']):
            weights.append(np.array(g['param_{}'.format(p)],dtype='float32'))
    f.close()
    return weights
    
def build_rnn(in_dim, out_dim, h0_dim, h1_dim=None, layer_type=SimpleRNN, return_sequences=True):
    model = Sequential()  
    model.add(layer_type(h0_dim, activation='relu',\
    input_dim=in_dim, return_sequences=return_sequences))  
    if h1_dim is not None:
        model.add(TimeDistributedDense(h1_dim))
        #model.add(layer_type(h1_dim, weights=,activation='relu',return_sequences=return_sequences))
    if return_sequences:
        model.add(TimeDistributedDense(out_dim))
    else:
        model.add(Dense(out_dim))  
    model.add(Activation("linear"))
    def mycost(y_true, y_pred):
        y_roll=T.roll(y_pred,1,axis=1)
        y_roll=T.set_subtensor(y_roll[:,0,:],y_pred[:,0,:])
        return T.mean(T.square(y_pred - y_true), axis=-1)+ 10*T.mean(T.square(y_pred - y_roll), axis=-1) 
    model.compile(loss=mycost, optimizer="rmsprop")  
    return model
    
def train(X_train, y_train, X_test, y_test, model, batch_size=128, nb_epoch=50):
    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    path=parent_path+"/RNN_weights"+tstr+".hdf5"
    checkpointer = ModelCheckpoint(filepath=path, verbose=1, save_best_only=True)
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              nb_epoch=nb_epoch, 
              show_accuracy=False, 
              verbose=2, 
              validation_data=(X_test, y_test), 
              callbacks=[early_stopping, checkpointer])
    
def build_rnn_dataset(steps):
    rnntrain,rnntest=makernnset(steps)
    print "trainset.shape, testset.shape =", rnntrain.shape, rnntest.shape
    X_train, y_train = np.dsplit(rnntrain,[5])
    X_valid, y_valid = np.dsplit(rnntest,[5])
                                           
    X_train = normalize(X_train)
    X_valid = normalize(X_valid)
    print 'X_train.shape, y_train.shape =', X_train.shape, y_train.shape
    return X_train, y_train, X_valid, y_valid
    
def normalize(X_train):
    assert X_train.ndim in [2,3], 'X_train dim wrong'
    if X_train.ndim==2:
        X_train = X_train-np.array([[0.,0.,1.9,0.,2.]])
        X_train = X_train/np.array([[7.7,10.0,1,0.03,0.5]])
    if X_train.ndim==3:
        X_train = X_train-np.array([[[0.,0.,1.9,0.,2.]]])
        X_train = X_train/np.array([[[7.7,10.0,1,0.03,0.5]]])
    return X_train.astype('float32')
    
def test(array,model):
    array=normalize(array)
    return model.predict(array[None,:],array.shape[0])[0,:,0]
    
def visual_test(model,sequence=None):
    parent_path=os.path.split(os.path.realpath(__file__))[0]
    if sequence==None:
        f = gzip.open(parent_path+'/dataset/sequence.pkl.gz', 'rb')
        sequence=cPickle.load(f)
        f.close()
    targets=sequence[:,5]
    predictions=test(sequence[:,:5],model)
    plt.figure('test')
    plt.title('test')
    plt.plot(targets,label='turn angle out')
    #plt.plot(3.*sequence[:,0],label='lean angle')
    #plt.plot(3.*sequence[:,1],label='lean angle rate')
    plt.plot(predictions,label='offline network out')
    plt.grid()
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.xlabel('time(20ms)')
    plt.ylabel('steering angle (degree)')
    plt.show()
    print 'abs error',np.mean(np.abs(predictions-targets))
    return predictions
        
    
def convert_norm(X):
    return X*np.array([[7.7,10.0,1,0.03,0.5]])+np.array([[0.,0.,1.9,0.,2.]])

if __name__=='__main__':
    parent_path = os.path.split(os.path.realpath(__file__))[0]
    X_train, y_train, X_valid, y_valid = build_rnn_dataset(10)
    RNNmodel=build_rnn(X_train.shape[-1] ,y_train.shape[-1], 100,20)
    
    #是否加载pretrain model
    #preweights=load_weights(parent_path+'/MLP_weightsMultispeed151226.hdf5')
    #preweights.insert(1,np.zeros((100,100),dtype='float32'))#recurrent weight
    #RNNmodel.set_weights(preweights)
    RNNmodel.load_weights(parent_path+'/RNN_weights10REGU1.0.hdf5')
    #train(X_train, y_train, X_valid, y_valid, RNNmodel, batch_size=128)
    #下面这个函数里可以选择可视化哪个数据文件
    testseq=loadgz('/Users/subercui/Git/BaikeBalance/RNN/dataset/testset.pkl.gz')
    predictions=visual_test(RNNmodel,testseq[6000:8000])
    #visual_test(RNNmodel,testseq)
    

'''可以调steps,optimizer'''    

