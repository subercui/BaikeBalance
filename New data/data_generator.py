# -*- coding: utf-8 -*-
'''
datagenerator:read datafiles and generate the dataset in a numpy matrix
现在这里也做了很多test
author: CUI HAOTIAN
date:2015.12.05
'''
import numpy as np
from matplotlib import pyplot as plt
import cPickle, gzip,os,re
import load_test as lt
import controlfunction as cf
from datetime import datetime
today=datetime.today()
tstr=today.strftime('%y%m%d')

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
path=parent_path+'/nn_test/'

def construct(matchs):
    data=[]
    for entry in matchs:
        print entry
        data.append(np.loadtxt(path+entry,delimiter=' '))
    data=np.vstack(data)
    #2.lean angle/degree 3.lean angle rate/degree 10.U0 11.turn angle in/rad 13.velocity 20.turn angle out
    data=data[:,(2,3,10,11,13,20)]    
    return data

def select(data):
    #1.筛选倾斜角较平稳的：（-7.7，+7.7）
    index=np.nonzero(np.abs(data[:,0])<7.7)
    data=data[index]
    #2.筛选倾斜角速度在（-10.0，+10.0）
    index=np.nonzero(np.abs(data[:,1])<10.)
    data=data[index]
    #3.筛选U0>=1.9
    index=np.nonzero(data[:,2]>=1.9)
    data=data[index]
    #4.筛选velocity在（1.5，2.5）
    index=np.nonzero(np.abs(data[:,4]-2)<0.5)
    data=data[index]
    #结果6个数据范围(-7.7,7.7),(-10,10),1.9,(-0.03,0.03),(1.5,2.5),(-30,+30)
    return data
    
def add_simu_data(data,length,rangelow=[-0.372,-0.859,1.9,0.,1.9],rangehigh=[0.372,0.859,1.9,0.,2.1]):
    #simu
    array=np.random.random((length,len(rangelow)))
    array=(np.array(rangehigh)-np.array(rangelow))*array+np.array(rangelow)
    pred=cf.control_func(array,array.shape[0])[:,None]
    array=np.hstack((array,pred))
    #combine
    data=np.vstack((data,array))
    data=np.roll(data,array.shape[0]*4/5,axis=0)
    return data
    

def savefile(m,path):
    save_file = gzip.open(path, 'wb')  # this will overwrite current contents
    cPickle.dump(m, save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()
    print 'saved at '+path
    
def visualize(data):
    n_sub=data.shape[1]
    rows=3
    lines=np.ceil(float(n_sub)/3)
    plt.figure('plot data')
    namelist=['lean angle(o)','lean angle rate(o/s)','U0',
    'turn angle in(rad)','velocity','turn angle out','nn turn angle out']
    for i in range(n_sub):
        plt.subplot(rows,lines,i+1)
        plt.plot(data[:,i], label=namelist[i], linewidth=1)
        plt.xlabel('index')
        plt.ylabel('Value')
        plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.show()

#读取记录数据，调查控制信号延迟    
def test_time_decay():
    data=np.loadtxt('/Volumes/NO NAME/nn_test/test_file_2015-12-5_17-36-33.txt',delimiter=' ')
    #2.lean angle/degree 3.lean angle rate/degree 10.U0 11.turn angle in/rad 13.velocity 20.turn angle out 21.nn turn angle out
    data=data[:,(1,2,3,10,11,13,20,21)]
    #data=select(data)
    #visualize(data)
    predictions=lt.test(data[:,1:6])
    plt.figure('test decay')
    plt.plot(data[:,-2],label='turn angle out')
    plt.plot(data[:,-1],label='network out')
    plt.plot(-data[:,1],label='steering filted')
    #plt.plot(predictions,label='offline network out')
    plt.grid()
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.xlabel('time(10ms)')
    plt.ylabel('Value')
    plt.show()

    return data 

if __name__== '__main__':
    '''matchs=filesinroot(path,".txt",0)
    data=construct(matchs)
    data=select(data)
    data=add_simu_data(data,data.shape[0]/2)
    visualize(data)
    savefile(data,parent_path+'/dataset/dataset'+tstr+'.pkl.gz')'''
    
    #test decay mode
    data=test_time_decay()
    
