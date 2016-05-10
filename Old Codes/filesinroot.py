import os

def filesinroot(dir,wildcard,recursion):
    matchs=[]
    exts=wildcard.split()
    for root,subdirs,files in os.walk(dir):
        for name in files:
            for ext in exts:
                if(name.endswith(ext)):
                    matchs.append(name)
                    break
        if(not recursion):
            break
    return matchs
path='./DataCut/'
matchs=filesinroot(path,".txt",0)

import numpy as np
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
        dataset[cnt,:,:]=entry[i*40:(i+1)*40,:]
        cnt=cnt+1
        
import gzip,cPickle
f=gzip.open('dataset.pkl.gz','wb')
cPickle.dump(dataset,f,-1)
f.close()
    