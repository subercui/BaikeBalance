# -*- coding: utf-8 -*-
#加入对样本长度（plot length-error），epoch次数（plot epoch-error），真实数据测试
#（plot 预测，真实）等的测试，对比RNN和MLP；展示以上内容

#调试神经元类型，数目，回溯结构等，√分开不同训练环境,修改分类问题，√输入3.0处理
import theano, theano.tensor as T
import numpy as np
from matplotlib import pyplot as plt
import theano_lstm
import random,re
import cPickle, gzip
from datetime import datetime
from theano_lstm import LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss
theano.config.compute_test_value = 'off'
theano.config.floatX = 'float32'
theano.config.mode='FAST_RUN'
theano.config.profile='False'
theano.config.scan.allow_gc='False'
#theano.config.device = 'gpu'

today=datetime.today()
def augment(data):
    #part0=data[:,:,]
    return data

def softmax(x):
    """
    Wrapper for softmax, helps with
    pickling, and removing one extra
    dimension that Theano adds during
    its exponential normalization.
    """
    return T.nnet.softmax(x.T)
    
def has_hidden(layer):
    """
    Whether a layer has a trainable
    initial hidden state.
    """
    return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
    return T.repeat(T.shape_padleft(vector), n, axis=0)

def initial_state(layer, dimensions = None):
    """
    Initalizes the recurrence relation with an initial hidden state
    if needed, else replaces with a "None" to tell Theano that
    the network **will** return something, but it does not need
    to send it to the next step of the recurrence
    """
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None
    
def initial_state_with_taps(layer, dimensions = None):
    """Optionally wrap tensor variable into a dict with taps=[-1]"""
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None

def create_shared(out_size, in_size=None, name=None):
    """
    Creates a shared matrix or vector
    using the given in_size and out_size.

    Inputs
    ------

    out_size int            : outer dimension of the
                              vector or matrix
    in_size  int (optional) : for a matrix, the inner
                              dimension.

    Outputs
    -------

    theano shared : the shared matrix, with random numbers in it

    """

    if in_size is None:
        return theano.shared(np.zeros((out_size, ),dtype=theano.config.floatX), name=name)
    else:
        return theano.shared(np.zeros((out_size, in_size),dtype=theano.config.floatX), name=name)

        
class Model:
    """
    Simple predictive model for forecasting words from
    sequence using LSTMs. Choose how many LSTMs to stack
    what size their memory should be, and how many
    words can be predicted.
    """
    def __init__(self, hidden_size, input_size, output_size, stack_size=1, celltype=Layer):
        # declare model
        self.model = StackedCells(input_size, celltype=celltype, layers =[hidden_size] * stack_size)
        # add a classifier:
        self.regression=Layer(hidden_size, output_size[0], activation = T.tanh)
        self.classifier=Layer(hidden_size, output_size[1], activation = softmax)
        # inputs are matrices of indices,
        # each row is a sentence, each column a timestep
        self.steps=T.iscalar('steps')
        self.x=T.tensor3('x')#输入gfs数据
        self.target0=T.tensor3('target0')#输出的目标target，这一版把target维度改了
        self.target1=T.itensor3('target1')
        self.layerstatus=None
        self.results=None
        # create symbolic variables for prediction:(就是做一次整个序列完整的进行预测，得到结果是prediction)
        self.predictions0,self.predictions1 = self.create_prediction()
        # create gradient training functions:
        self.create_cost_fun()
        self.create_valid_error()
        self.create_training_function()
        self.create_predict_function()
        self.create_validate_function()
        '''上面几步的意思就是先把公式写好'''
        
        
    @property
    def params(self):
        return self.model.params+self.regression.params+self.classifier.params
        
    def create_prediction(self):#做一次predict的方法        
        def step(idx):
            new_states=self.model.forward(idx)
            output0=self.regression.activate(new_states[-1])
            output1=self.classifier.activate(new_states[-1])
            return [output0,output1]#不论recursive与否，会全部输出
        
        x = self.x
        num_examples = x.shape[0]
        #outputs_info =[initial_state_with_taps(layer, num_examples) for layer in self.model.layers]
        #outputs_info = [initial_state_with_taps(layer, num_examples) for layer in self.model.layers[1:]]
        [result0,result1], _ = theano.scan(fn=step,
                                n_steps=self.steps,
                                sequences=dict(input=x.dimshuffle((1,0,2)), taps=[-0]),
                                )
                                

        return result0.dimshuffle((1,0,2)),result1.dimshuffle((2,0,1))
        
        
    def create_cost_fun (self):
        y=self.target1[:,0,0]                                 
        self.cost = (self.predictions0 - self.target0[:,:,0:1]).norm(L=2)+10*(-T.mean(T.log(self.predictions1)[T.arange(y.shape[0]),:,y]))

    def create_valid_error(self):
        self.valid_error0=T.mean(T.abs_(self.predictions0 - self.target0[:,:,0:1]),axis=0)
        #self.valid_error1=-T.mean(T.log(self.predictions1)[T.arange(self.target1.shape[0]),:,self.target1[:,0,0]])
        self.valid_error1=T.mean(T.eq(T.argmax(self.predictions1, axis=2).dimshuffle(1,0),self.target1[:,0,0]))
                
    def create_predict_function(self):
        self.pred_fun = theano.function(inputs=[self.x,self.steps],outputs =[self.predictions0,self.predictions1],allow_input_downcast=True)
                                 
    def create_training_function(self):
        updates, gsums, xsums, lr, max_norm = create_optimization_updates(self.cost, self.params, method="adadelta")#这一步Gradient Decent!!!!
        self.update_fun = theano.function(
            inputs=[self.x, self.target0,self.target1,self.steps],
            outputs=self.cost,
            updates=updates,
            name='update_fun',
            profile=False,
            allow_input_downcast=True)
            
    def create_validate_function(self):
        self.valid_fun = theano.function(
            inputs=[self.x, self.target0,self.target1,self.steps],
            outputs=[self.valid_error0,self.valid_error1],
            allow_input_downcast=True
        )
        
    def __call__(self, x):
        return self.pred_fun(x)

#################
# GENERATE DATA #
#################
print '... generating data'
import os
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

def onecircle(setlength,setn_epochs):
    
    data=[]
    for entry in matchs:
        print entry
        data.append(np.loadtxt(path+entry,delimiter=','))
    
    length=setlength
    expcnt=0
    for entry in data:
        expcnt=expcnt+entry.shape[0]/length
    dataset=np.zeros((expcnt,length,8))  
    cnt=0 
    for entry in data:
        print entry.shape
        for i in range(entry.shape[0]/length):
            dataset[cnt,:,:]=entry[i*length:(i+1)*length,:]
            cnt=cnt+1
    data=dataset
    
    #############
    # LOAD DATA #
    #############
    '''print '... loading data'
    f=gzip.open('dataset.pkl.gz','rb')
    data=cPickle.load(f)'''
    data=np.asarray(data,dtype='float32')
    np.random.shuffle(data)
    paramin=np.amin(np.amin(data,axis=0),axis=0)[None,None,:]
    paramax=np.amax(np.amax(data,axis=0),axis=0)[None,None,:]
    data=(data-paramin)/(paramax-paramin)
    #categorize
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j,6]>0.9:
                data[i,j,6]=2
            elif data[i,j,6]>0.1 and data[i,j,6]<0.9:
                data[i,j,6]=1
            else:
                data[i,j,6]=0
    
    train_data,test_data=np.split(data,[int(0.8*data.shape[0])],axis=0)
    #augment
    train_data=augment(train_data)
    X_train,Y_train=np.split(train_data,[5],axis=2)
    X_test, Y_test=np.split(test_data,[5],axis=2)
                    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    steps=length
    RNNobj = Model(
        input_size=5,
        hidden_size=10,
        output_size=[1,3],
        stack_size=1, # make this bigger, but makes compilation slow
        celltype=Layer, # use RNN or LSTM
    )
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    
    batch=40
    train_batches=X_train.shape[0]/batch
    valid_batches=X_test.shape[0]/batch
    
    #a=RNNobj.pred_fun(X_train[batch*0:batch*(0+1)])
    
    train_error=np.zeros(setn_epochs)
    valid_error0=np.zeros(setn_epochs)
    valid_error1=np.zeros(setn_epochs)
    for k in xrange(setn_epochs):#run k epochs
        error_addup=0
        for i in xrange(train_batches): #an epoch
            error_addup=RNNobj.update_fun(X_train[batch*i:batch*(i+1)],Y_train[batch*i:batch*(i+1),:,0:1],np.floor(Y_train[batch*i:batch*(i+1),:,1:2]),setlength)+error_addup
        error=error_addup/(i+1)
        print ("epoch %(epoch)d,\n  train error=%(error)f" % ({"epoch": k+1, "error": error}))
        train_error[k]=error
        
        valid_error_addup0=0
        valid_error_addup1=0
        for i in xrange(valid_batches): #an epoch
            error0,error1=RNNobj.valid_fun(X_test[batch*i:batch*(i+1)],Y_test[batch*i:batch*(i+1),:,0:1],np.floor(Y_test[batch*i:batch*(i+1),:,1:2]),setlength)
            valid_error_addup0=error0+valid_error_addup0
            valid_error_addup1=error1+valid_error_addup1
        error0=valid_error_addup0[0,0]/(i+1)
        error1=valid_error_addup1/(i+1)
        print ("  validation error:%(error0).4f %(error1).4f\n"%({"epoch":k+1, "error0":error0, "error1":error1}))
        #print error
        valid_error0[k]=error0
        valid_error1[k]=error1
        #print valid_error_addup0/(i+1), valid_error_addup1/(i+1)
        
    
    return train_error,valid_error0,valid_error1,RNNobj

lengthlist=[1]
RNNobjlist=range(len(lengthlist))
n_epochs=200
train_error=np.zeros((len(lengthlist),n_epochs))
valid_error0=np.zeros((len(lengthlist),n_epochs))
valid_error1=np.zeros((len(lengthlist),n_epochs))
for i in range(len(lengthlist)):
    train_error[i],valid_error0[i],valid_error1[i],RNNobjlist[i]=onecircle(lengthlist[i],n_epochs)
    #############
    # VISUALIZE #
    #############
    plt.figure('Training Error %d'%i)
    plt.plot(train_error[i],'g-', label='train error', linewidth=2)
    plt.title('Training Error - Epochs For Length: %d'%(lengthlist[i]))
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.legend()
    '''
    plt.figure('Total Valid Error %d'%i)
    plt.plot(valid_error[i], label='valid error', linewidth=2)
    plt.title('Valid Error - Epochs For Length: %d'%(lengthlist[i]))
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.legend()
    '''
    plt.figure('Valid Error %d'%i)
    plt.subplot(2, 1, 1)
    plt.plot(valid_error0[i,:], label='turn angle', linewidth=2)
    plt.title('Valid Error - Epochs For Length: %d'%(lengthlist[i]))
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.subplot(2, 1, 2)
    plt.plot(valid_error1[i,:], label='brake', linewidth=2)
    #plt.plot(valid_error3[i,:,2], label='speed', linewidth=2)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.show()

'''    
plt.figure('length-train error')
plt.plot(lengthlist,train_error[:,-1],'go-', label='train error', linewidth=2)
plt.title('Training Error - Length at Epoch: %d'%(n_epochs))
plt.xlabel('Length')
plt.ylabel('error')
plt.legend(loc='best', fancybox=True, framealpha=0.5)

plt.figure('length-valid error')
plt.plot(lengthlist,valid_error[:,-1],'o-', label='mean valid error', linewidth=2)
plt.title('Valid Error - Length at Epoch: %d'%(n_epochs))
plt.xlabel('Length')
plt.ylabel('error')
plt.plot(lengthlist,valid_error3[:,-1,0],'o-', label='turn angle', linewidth=2)
plt.plot(lengthlist,valid_error3[:,-1,1],'o-', label='brake', linewidth=2)
plt.plot(lengthlist,valid_error3[:,-1,2],'o-', label='speed', linewidth=2)
plt.legend(loc='best', fancybox=True, framealpha=0.5)
plt.show()'''

    
###############
# Real TEST #
###############
print 'real test'
f=gzip.open(parent_path+'/dataset.pkl.gz','rb')
data=cPickle.load(f)
data=np.asarray(data,dtype='float32')
#np.random.shuffle(data)
paramin=np.amin(np.amin(data,axis=0),axis=0)[None,None,:]
paramax=np.amax(np.amax(data,axis=0),axis=0)[None,None,:]
data=(data-paramin)/(paramax-paramin)
#categorize
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i,j,6]>0.9:
            data[i,j,6]=2
        elif data[i,j,6]>0.1 and data[i,j,6]<0.9:
            data[i,j,6]=1
        else:
            data[i,j,6]=0
    
X,Y=np.split(data,[5],axis=2)
s=6    
a0,a1=RNNobjlist[0].pred_fun(X[s:s+1],40)
plt.figure('real curve-predict curve')
plt.plot(a0[0,:],'--', label='predict turn angle', linewidth=2)
plt.plot(np.argmax(a1,axis=2).flatten(),'--', label='predict brake', linewidth=2)
#plt.plot(a[0,:,2],'--', label='predict speed', linewidth=2)
plt.title('Real Sequence-Predict Sequence - Length 1 at Epoch: %d'%(n_epochs))
plt.xlabel('Sequence')
plt.ylabel('Value')
plt.plot(Y[s,:,0],'o-', label='real turn angle', linewidth=2)
plt.plot(Y[s,:,1],'o-', label='real brake', linewidth=2)
#plt.plot(Y[s,:,2],'o-', label='real speed', linewidth=2)
plt.legend(loc='best', fancybox=True, framealpha=0.5)
plt.show()
    
'''##############
# SAVE MODEL #
##############
savedir='/data/pm25data/model/RNNModel'+today.strftime('%Y%m%d')+'.pkl.gz'
save_file = gzip.open(savedir, 'wb')
cPickle.dump(RNNobj.model.params, save_file, -1)
cPickle.dump(para_min, save_file, -1)#scaling paras
cPickle.dump(para_max, save_file, -1)
save_file.close()

print ('model saved at '+savedir)'''


'''
记录：
一.超参数表：网络方面：单层神经元数，batch大小，神经元类型，scaling(似乎没影响)
学习方面：lr,costfunction中比例α
1.网络（5，10，1+3），batch40，tanh，α=10得到
error=2.112704 valid[[ 0.04164371]] 0.950704225352
2.网络（5，10，1+3），batch40，tanh，α=5得到
epoch 200, error=1.216951 valid[[ 7.822875%]] 0.95
3.网络（5，10，1+3），batch40，tanh，α=100得到
从epoch195开始震荡，train loss上升，不对了
'''
