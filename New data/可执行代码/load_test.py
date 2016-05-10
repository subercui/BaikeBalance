'''
load&predict 
author: CUI HAOTIAN
date:2015.12.05
'''
import modelfunction as mf
import cPickle, gzip,os
import numpy as np
from utils import loadgz
MLPmodel = None

def init(weight='MLP_weightsBest.hdf5'):
    print '... building the model'
    global MLPmodel
    if MLPmodel==None:
        MLPmodel=mf.build_mlp(5,1, 100,20)
        MLPmodel.path=weight
        parent_path = os.path.split(os.path.realpath(__file__))[0]
        MLPmodel.load_weights(parent_path+'/'+weight)
    if not MLPmodel.path==weight:
        MLPmodel=mf.build_mlp(5,1, 100,20)
        MLPmodel.path=weight
        parent_path = os.path.split(os.path.realpath(__file__))[0]
        MLPmodel.load_weights(parent_path+'/'+weight)
    
    return MLPmodel
    

def mean_square_error(predictions, targets):
    return np.square(predictions - targets).mean(axis=0)

def absolute_percent_error(predictions, targets, targets_mean):
    return (np.abs(predictions - targets) / np.abs(targets_mean)).mean(axis=0)
        
def absolute_error(predictions, targets):
    return np.abs(predictions - targets).mean(axis=0)
    
def test(array,weightfile='MLP_weightsMultispeed151226.hdf5'):
    #array=np.array([[ 0.09167325, 0.006      ,0,0,0]])
    MLPmodel=init(weight=weightfile)
    array=mf.normalize(array)
    return MLPmodel.predict(array,array.shape[0])
    
if __name__=='__main__':
    parent_path = os.path.split(os.path.realpath(__file__))[0]
    array=np.array([[0.,20.,1.9,0.,2.0]])
    print test(array)
    testset=loadgz(parent_path+'/dataset/testset.pkl.gz')
    mf.visual_test(MLPmodel,sequence=testset[6000:8000])    
        