# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from utils import loadgz

def sigment(data,thr):
    tmp=np.abs(data[:,0]-np.roll(data[:,0],1))
    #plt.plot(tmp)
    #plt.show()
    starts=np.nonzero(tmp>thr)
    print 'data segments',starts
    #本帧和上一帧差值过大，这应该是另一个sequence的起点
    return starts

def makedata(data,starts,steps):
    starts=starts[0].tolist()
    starts.insert(0,0)
    starts.append(-1)
    rnndata=[]
    for i in range(len(starts)-1):
        period=data[starts[i]:starts[i+1],:]
        #print 'iter #',i
        for idx in range(period.shape[0]-steps):
            rnndata.append(period[idx:idx+steps,:])
    rnndata=np.array(rnndata)
    return rnndata

def makernnset(steps):
    train=loadgz('/Users/subercui/Git/BaikeBalance/RNN/dataset/trainset.pkl.gz')
    test=loadgz('/Users/subercui/Git/BaikeBalance/RNN/dataset/testset.pkl.gz')
                
    trainstarts=sigment(train,4)
    teststarts=sigment(test,1.5)
    
    rnntrain=makedata(train,trainstarts,steps)
    rnntest=makedata(test,teststarts,steps)
    return rnntrain,rnntest
    
if __name__=='__main__':
    rnntrain,rnntest=makernnset(10)        