# -*- coding: utf-8 -*-
import theano, theano.tensor as T
import numpy as np
from matplotlib import pyplot as plt
import theano_lstm
import random,re,os,time,serial,math
import cPickle, gzip
from theano_lstm import LSTM, RNN, StackedCells, Layer, create_optimization_updates
pi=math.pi

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
    def __init__(self, hidden_size, input_size, output_size, celltype=Layer):
        # declare model
        self.model = StackedCells(input_size, celltype=celltype, layers =hidden_size)
        # add a classifier:
        self.regression=Layer(hidden_size[-1], output_size[0], activation = T.tanh)
        self.classifier=Layer(hidden_size[-1], output_size[1], activation = softmax)
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
        #self.create_cost_fun()
        #self.create_valid_error()
        #self.create_training_function()
        self.create_predict_function()
        #self.create_validate_function()
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
        self.cost = (self.predictions0 - self.target0[:,:,0:1]).norm(L=2)+100*(-T.mean(T.log(self.predictions1)[T.arange(y.shape[0]),:,y]))

    def create_valid_error(self):
        self.valid_error0=T.mean(T.abs_(self.predictions0 - self.target0[:,:,0:1]),axis=0)
        #self.valid_error1=-T.mean(T.log(self.predictions1)[T.arange(self.target1.shape[0]),:,self.target1[:,0,0]])
        self.valid_error1=T.mean(T.eq(T.argmax(self.predictions1, axis=2).dimshuffle(1,0),self.target1[:,0,0]))
                
    def create_predict_function(self):
        self.pred_fun = theano.function(inputs=[self.x,self.steps],outputs =[self.predictions0,self.predictions1],allow_input_downcast=True)
                                 
    def create_training_function(self):
        updates, gsums, xsums, lr, max_norm = create_optimization_updates(self.cost, self.params, lr=0.01, method="adagrad")#这一步Gradient Decent!!!!
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

def init(length,n_hiddens):
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    steps=length
    RNNobj = Model(
        input_size=5,
        hidden_size=n_hiddens,
        output_size=[1,3],
        celltype=Layer, # use RNN or LSTM
    )
    
    return RNNobj
    
def load_weight(RNNobj):
    parent_path = os.path.split(os.path.realpath(__file__))[0]
    path=parent_path+'/BikeModel.pkl.gz'
    f=gzip.open(path, 'rb')
    #f=gzip.open('DetachValidModel20150901.pkl.gz', 'rb')
    #for i in range(len(RNNobj.model.layers)):
    #    RNNobj.model.layers[i].params=cPickle.load(f)
    params=cPickle.load(f)
    para_min=cPickle.load(f)
    para_max=cPickle.load(f)
    f.close()
    
    return params,para_min,para_max

def hexShow(argv):  
    result = ''  
    hLen = len(argv)  
    for i in xrange(hLen):  
        hvol = ord(argv[i])  
        hhex = '%02x'%hvol  
        result += hhex  
    print 'hexShow:',result
    return result

def genOutput(a,RNNobj,para_min,para_max):
    if len(a)>=44:
        #检验起始FF
        if int(a[0:2], 16)!=255:
            print 'missing header ff'
            #raise Exception,'missing header ff'
        if int(a[4:6], 16)!=23:
            print 'wrong length flag'
            #raise Exception,'wrong length flag'
        if int(a[6:8], 16)!=2:
            print 'wrong length'
            #raise Exception,'wrong length flag'
        #decode
        speed=int(a[8:12], 16)#两字节车速
        lean=int(a[12:16], 16)#
        leanspeed=int(a[16:20], 16)
        U0=int(a[40:44], 16)
        turncontrol=int(a[44:48], 16)
        speed=speed/1000.
        lean=(lean/10000.-1.)*180/pi
        leanspeed=leanspeed/10000.-1
        U0=U0/1000.
        turncontrol=(turncontrol/10000.-1.)*180/pi
        inputs=np.array([[[lean,leanspeed,turncontrol,speed,U0]]], dtype='float32')
        #inputs scale
        inputs=(inputs-para_min[:,:,:5])/(para_max[:,:,:5]-para_min[:,:,:5])
        #predict
        result=RNNobj.pred_fun(inputs,1)
        #outputs scale and encode
        turnangle=result[0][0,0,0]*(para_max[0,0,0]-para_min[0,0,0])+para_min[0,0,0]
        turnangle=int((turnangle/180*pi+1)*10000)
        brake=np.argmax(result[1])
        if brake>=2:
            U=int(U0*1000)
            brake=75
        elif brake==1:
            U=int(1.5*1000)
            brake=68
        else:
            U=int(1.5*1000)
            brake=62
        U="%04x"%U
        turnangle="%04x"%turnangle
        brake="%02x"%brake
        tosend='ff000702'+U+turnangle+brake
        #^ check
        check=int(tosend[0:2],16)
        for i in xrange(2,18,2):
            check=check^int(tosend[i:i+2],16)
        tosend=tosend+"%02x"%check
    else:
        tosend=''
    
    return tosend,result
        


    
if __name__== '__main__':
    length=1
    n_hiddens=[100]
    RNNobj=init(length,n_hiddens)
    params,para_min,para_max=load_weight(RNNobj)
    for i in xrange(len(RNNobj.params)):
        RNNobj.params[i].set_value(params[i].eval())
    '''#ser = serial.Serial('/dev/ttyUSB0',115200,timeout=0)
    while(1):
        now=time.time()
        #print 'read serial'
        #a=ser.read(27) 
        #print 'readout',hexer
        #a=hexShow(a)
        a='ff0017020000246127147c2fd239263e25824d7800002710000099'
        tosend,result=genOutput(a,RNNobj,para_min,para_max)
        #a=np.array([[[ 0.44103992,  0.51292008,  0.5, 0.77564108,  0.79999977]]], dtype='float32')
        #tosend='2710'
        hexsend=tosend.decode('hex')
        #ser.write(hexsend)
        #print 1000*(time.time()-now),'ms'
        while time.time()-now<0.01:
            continue
        print 1000*(time.time()-now),'ms'
    #ser.close()'''


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
theme="straight"
matchs=filesinroot(path,theme,0)
data=[]
for entry in matchs:
    print entry
    data.append(np.loadtxt(path+entry,delimiter=','))
    
length=40
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
#f=gzip.open(parent_path+'/dataset.pkl.gz','rb')
#data=cPickle.load(f)
data=np.asarray(data,dtype='float32')
np.random.shuffle(data)
dataorigin=np.array(data)
data=(data-para_min)/(para_max-para_min)
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
for s in range(6,20):    
    a0,a1=RNNobj.pred_fun(X[s:s+1],40)
    plt.figure('curve-predict for '+theme+' curve index %d'%s,figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(X[s,:,0],'o-', label='lean angle', linewidth=2)
    plt.plot(X[s,:,1],'o-', label='lean angle speed', linewidth=2)
    plt.plot(X[s,:,2],'o-', label='turn control', linewidth=2)
    plt.plot(X[s,:,3],'o-', label='real speed', linewidth=2)
    plt.plot(X[s,:,4],'o-', label='speed voltage', linewidth=2)
    plt.title('Input Sequence - Length 1 at Epoch: ')
    plt.xlabel('Sequence')
    plt.ylabel('Value')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.subplot(1, 2, 2)
    plt.plot(a0[0,:]*(para_max[:,:,5]-para_min[:,:,5])+para_min[:,:,5],'^-', label='predict turn angle', linewidth=2)
    #plt.plot(np.argmax(a1,axis=2).flatten(),'^-', label='predict brake', linewidth=2)
    #plt.plot(a[0,:,2],'--', label='predict speed', linewidth=2)
    plt.title('Real Sequence-Predict Sequence - Length 1 at Epoch: ')
    plt.xlabel('Sequence')
    plt.ylabel('Value')
    plt.plot(dataorigin[s,:,5],'o-', label='real turn angle', linewidth=2)
    #plt.plot(Y[s,:,1],'o-', label='real brake', linewidth=2)
    #plt.plot(Y[s,:,2],'o-', label='real speed', linewidth=2)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.show()

predict=RNNobj.pred_fun(X,40)[0][:,:,0]*(para_max[:,:,5]-para_min[:,:,5])+para_min[:,:,5]
target=dataorigin[:,:,5]

def showstats():
    print 'predict mean',np.abs(predict).mean()
    print 'predict max',np.abs(predict).max()
    print 'target mean',np.abs(target).mean()
    print 'target max',np.abs(target).max()
    print 'minus mean',np.abs(predict-target).mean()
    print 'minus max',np.abs(predict-target).max()

def velocitytest():
    test=np.array([[[ 0.55,  0.495,  0.5, 0.77564108,  0.]]], dtype='float32')
    print test*(para_max[:,:,:5]-para_min[:,:,:5])+para_min[:,:,:5]
    testUplus=np.array([[[ 0.44103992,  0.51292008,  0.5, 0.77564108,  0.79999977]]], dtype='float32')
    testVplus=np.array([[[ 0.44103992,  0.51292008,  0.5, 0,  0.79999977]]], dtype='float32')
    print 'test',(RNNobj.pred_fun(test,1)[0][:,:,0]*(para_max[:,:,5]-para_min[:,:,5])+para_min[:,:,5])[0,0]
    print 'testUplus',(RNNobj.pred_fun(testUplus,1)[0][:,:,0]*(para_max[:,:,5]-para_min[:,:,5])+para_min[:,:,5])[0,0]
    print 'testVplus',(RNNobj.pred_fun(testVplus,1)[0][:,:,0]*(para_max[:,:,5]-para_min[:,:,5])+para_min[:,:,5])[0,0]

def rawinputtest():
    test=np.array([[[-3.99351583206, 0.0026, 0.0, 0.0, 1.9]]],dtype='float32')
    test=(test-para_min[:,:,:5])/(para_max[:,:,:5]-para_min[:,:,:5])
    print 'test',test
    toprint=RNNobj.pred_fun(test,1)[0][0,0,0]
    print 'result',toprint
    print toprint*(para_max[0,0,5]-para_min[0,0,5])+para_min[0,0,5]
    