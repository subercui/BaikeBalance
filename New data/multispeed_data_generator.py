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
import data_generator as dg
from datetime import datetime
import utils
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
        data.append(np.loadtxt(path+entry))
    data=np.vstack(data)
    #2.lean angle/degree 3.lean angle rate/degree 10.U0 11.turn angle in/rad 13.velocity 20.turn angle out
    data=data[:,(1,2,9,10,12,18)]    
    return data

def select(data):
    #1.筛选倾斜角较平稳的：（-7.7，+7.7）
    #index=np.nonzero(np.abs(data[:,0])<7.7)
    #data=data[index]
    #2.筛选倾斜角速度在（-10.0，+10.0）
    index=np.nonzero(np.abs(data[:,1])<20.)
    data=data[index]
    #3.筛选U0>=1.9
    index=np.nonzero(data[:,2]>=1.9)
    data=data[index]
    index=np.nonzero(data[:,2]<=3)
    data=data[index]
    #4.筛选velocity在（1.5，2.5）
    #index=np.nonzero(np.abs(data[:,4]-2)<0.5)
    #data=data[index]
    #结果6个数据范围(-7.7,7.7),(-20,20),1.9,(-0.03,0.03),(1.5,2.5),(-30,+30)
    return data

#加模拟数据    
def add_simu_data(data,length,rangelow=[-0.372,-0.859,1.9,0.,1.9],rangehigh=[0.372,0.859,2.2,0.,200]):
    #simu
    array=np.random.random((length,len(rangelow)))
    array=(np.array(rangehigh)-np.array(rangelow))*array+np.array(rangelow)
    pred=cf.control_func(array,array.shape[0])[:,None]
    array=np.hstack((array,pred))
    #combine
    data=np.vstack((data,array))
    data=np.roll(data,array.shape[0]*4/5,axis=0)
    #我一直改的上面这一句，发现4/5，比较好，本来是3/4感觉拿到数据容易过拟合，那边取了7/8
    #其实就是那边后边0的部分留的太多，validation就会小很多，误以为好
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
    plt.figure('plot multispeed data')
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
def test_time_decay(path='/Users/subercui/Git/BaikeBalance/New data/nn_test/NNcontrol_2015-12-20_11-39-36.txt'):
    #data=np.loadtxt('/Volumes/NO NAME/nn_test/test_file_2015-12-5_17-36-33.txt',delimiter=' ')
    data=np.loadtxt(path)
    print data.shape
    #2.lean angle/degree 3.lean angle rate/degree 10.U0 11.turn angle in/rad 13.velocity 18.turn angle out 19.nn turn angle out
    data=data[:,(1,2,9,10,12,18,19)]
    #data=select(data)
    #visualize(data)
    predictions=lt.test(data[:,0:5],weightfile='MLP_weightsBest.hdf5')
    predictions2=lt.test(data[:,0:5],weightfile='MLP_weightsMultispeed151226.hdf5')
    #simupred=cf.control_func(data[:,0:5],data[:,0:5].shape[0])
    plt.figure('test decay')
    plt.plot(data[:,-2],label='turn angle out')
    plt.plot(data[:,-1],label='network out')
    #plt.plot(simupred[:],label='simulation out')
    plt.plot(predictions[:],label='offline network out')
    plt.plot(predictions2[:],label='offline mutispeed')
    plt.grid()
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.xlabel('time(10ms)')
    plt.ylabel('Value')
    plt.show()

    return data 

if __name__== '__main__':

    matchs=filesinroot(path,"multispeed",0)
    multispeed_data=construct(matchs)
    multispeed_data=select(multispeed_data)
    #multispeed_data=add_simu_data(multispeed_data,multispeed_data.shape[0]/2)
    visualize(multispeed_data)
    savefile(multispeed_data,parent_path+'/dataset/multispeed_dataset'+tstr+'.pkl.gz')

    '''
    matchs=filesinroot(path,"test",0)
    data=dg.construct(matchs)
    data=dg.select(data)
    data=dg.add_simu_data(data,data.shape[0]/2)
    data=np.vstack((multispeed_data,data))
    dg.visualize(data)
    savefile(multispeed_data,parent_path+'/dataset/dataset'+tstr+'.pkl.gz')
    '''
    '''
    #test decay mode
    data=test_time_decay(path='/Users/subercui/Git/BaikeBalance/New data/nn_test/U2.0_2016-1-16_11-15-8.txt')
    utils.visualize(data)'''
