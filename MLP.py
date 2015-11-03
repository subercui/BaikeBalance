# -*- coding: utf-8 -*-
#加入对样本长度（plot length-error），epoch次数（plot epoch-error），真实数据测试
#（plot 预测，真实）等的测试，对比RNN和MLP；展示以上内容
import theano, theano.tensor as T
import numpy as np
import theano_lstm
import random
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
    def __init__(self, hidden_size, input_size, output_size, stack_size=1, celltype=RNN,steps=40):
        # declare model
        self.model = StackedCells(input_size, celltype=celltype, layers =[hidden_size] * stack_size)
        # add a classifier:
        self.model.layers.append(Layer(hidden_size, output_size, activation = T.tanh))
        # inputs are matrices of indices,
        # each row is a sentence, each column a timestep
        self.steps=steps
        self.x=T.tensor3('x')#输入gfs数据
        self.target=T.tensor3('target')#输出的目标target，这一版把target维度改了
        self.layerstatus=None
        self.results=None
        # create symbolic variables for prediction:(就是做一次整个序列完整的进行预测，得到结果是prediction)
        self.predictions = self.create_prediction()
        # create gradient training functions:
        self.create_cost_fun()
        self.create_valid_error()
        self.create_training_function()
        self.create_predict_function()
        self.create_validate_function()
        '''上面几步的意思就是先把公式写好'''
        
        
    @property
    def params(self):
        return self.model.params
        
    def create_prediction(self):#做一次predict的方法
        '''x=self.x
        #初始第一次前传
        self.layerstatus=self.model.forward(x[:,0])
	#results.shape?40*1
        self.results=self.layerstatus[-1].dimshuffle((0,'x',1))
        if self.steps > 1:
            for i in xrange(1,self.steps):
                self.layerstatus=self.model.forward(x[:,i],self.layerstatus)
                #need T.shape_padright???
                self.results=T.concatenate([self.results,self.layerstatus[-1].dimshuffle((0,'x',1))],axis=1)
        return self.results'''
        
        def step(idx,*states):
            newstates=list(states)
            new_states=self.model.forward(idx,prev_hiddens = newstates)
            return new_states#不论recursive与否，会全部输出
        
        x = self.x
        num_examples = x.shape[0]
        outputs_info =[initial_state_with_taps(layer, num_examples) for layer in self.model.layers]
        #outputs_info = [initial_state_with_taps(layer, num_examples) for layer in self.model.layers[1:]]
        result, _ = theano.scan(fn=step,
                                n_steps=self.steps,
                                sequences=dict(input=x.dimshuffle((1,0,2)), taps=[-0]),
                                outputs_info=outputs_info)
                                

        return result[-1].dimshuffle((1,0,2))
        
    def create_cost_fun (self):                                 
        self.cost = (self.predictions - self.target).norm(L=2)

    def create_valid_error(self):
        self.valid_error=T.mean(T.abs_(self.predictions - self.target),axis=0)
                
    def create_predict_function(self):
        self.pred_fun = theano.function(inputs=[self.x],outputs =self.predictions,allow_input_downcast=True)
                                 
    def create_training_function(self):
        updates, gsums, xsums, lr, max_norm = create_optimization_updates(self.cost, self.params, method="adadelta")#这一步Gradient Decent!!!!
        self.update_fun = theano.function(
            inputs=[self.x, self.target],
            outputs=self.cost,
            updates=updates,
            name='update_fun',
            profile=False,
            allow_input_downcast=True)
            
    def create_validate_function(self):
        self.valid_fun = theano.function(
            inputs=[self.x, self.target],
            outputs=self.valid_error,
            allow_input_downcast=True
        )
        
    def __call__(self, x):
        return self.pred_fun(x)

#############
# LOAD DATA #
#############
print '... loading data'
f=gzip.open('dataset.pkl.gz','rb')
data=cPickle.load(f)
data=np.asarray(data,dtype='float32')
np.random.shuffle(data)
paramin=np.amin(np.amin(data,axis=0),axis=0)[None,None,:]
paramax=np.amax(np.amax(data,axis=0),axis=0)[None,None,:]
data=(data-paramin)/(paramax-paramin)

train_data,test_data=np.split(data,[int(0.8*data.shape[0])],axis=0)
X_train,Y_train=np.split(train_data,[5],axis=2)
X_test, Y_test=np.split(test_data,[5],axis=2)
                
######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'
steps=40
RNNobj = Model(
    input_size=5,
    hidden_size=10,
    output_size=3,
    stack_size=1, # make this bigger, but makes compilation slow
    celltype=Layer, # use RNN or LSTM
    steps=steps
)

###############
# TRAIN MODEL #
###############
print '... training'

batch=30
train_batches=X_train.shape[0]/batch
valid_batches=X_test.shape[0]/batch

a=RNNobj.pred_fun(X_train[batch*0:batch*(0+1)])

for k in xrange(100):#run k epochs
    error_addup=0
    for i in xrange(train_batches): #an epoch
    #for i in xrange(100): #an epoch
        error_addup=RNNobj.update_fun(X_train[batch*i:batch*(i+1)],Y_train[batch*i:batch*(i+1)])+error_addup
        if i%(train_batches/3) == 0:
	    error=error_addup/(i+1)
            print ("batch %(batch)d, error=%(error)f" % ({"batch": i+1, "error": error}))
    error=error_addup/(i+1)
    print ("   epoch %(epoch)d, error=%(error)f" % ({"epoch": k+1, "error": error}))
    
    valid_error_addup=0
    for i in xrange(valid_batches): #an epoch
    #for i in xrange(100):
        valid_error_addup=RNNobj.valid_fun(X_test[batch*i:batch*(i+1)],Y_test[batch*i:batch*(i+1)])+valid_error_addup
        if i%(valid_batches/3) == 0:
            #error=valid_error_addup/(i+1)
	    print ("batch %(batch)d, validation error:"%({"batch":i+1}))
            #print error
            #print ("batch %(batch)d, validation error=%(error)f" % ({"batch": i, "error": error}))
    error=np.mean(valid_error_addup/(i+1),axis=0)
    print ("epoch %(epoch)d, validation error:%(error)f"%({"epoch":k+1, "error":np.mean(error)}))
    print error
    #print ("   validation epoch %(epoch)d, validation error=%(error)f" % ({"epoch": k, "error": error}))

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
