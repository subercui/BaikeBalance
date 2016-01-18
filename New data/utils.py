# -*- coding: utf-8 -*-
'''
utils functions
'''
from matplotlib import pyplot as plt
import cPickle, gzip,os
import numpy as np

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
    #结果6个数据范围(-7.7,7.7),(-10,10),1.9,(-0.03,0.03),(1.5,2.5),(-30,+30)
    return data

def visualize(data):
    n_sub=data.shape[1]
    rows=3
    lines=np.ceil(float(n_sub)/3)
    plt.figure('plot data content')
    namelist=['lean angle(o)','lean angle rate(o/s)','U0',
    'turn angle in(rad)','velocity','turn angle out','nn turn angle out']
    for i in range(n_sub):
        plt.subplot(rows,lines,i+1)
        plt.plot(data[:,i], label=namelist[i], linewidth=1)
        plt.xlabel('index')
        plt.ylabel('Value')
        plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.show()

def loadgz(path):
    f = gzip.open(path, 'rb')  
    data = cPickle.load(f)
    f.close()
    return data
    
def combine(data):
    n_data=len(data)
    assert n_data>1, 'need at least 2 data files'
    combined=np.concatenate(data,axis=0)
    return combined
    
def savefile(m,path):
    save_file = gzip.open(path, 'wb')  # this will overwrite current contents
    cPickle.dump(m, save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()
    print 'saved at '+path
    
def construct(f):
    data=np.loadtxt(f)
    #2.lean angle/degree 3.lean angle rate/degree 10.U0 11.turn angle in/rad 13.velocity 20.turn angle out
    data=data[:,(1,2,9,10,12,18)]    
    return data
    
#give filenames generate 6 dimention datas
def makedatas(*files):
    n_files=len(files)
    data=[]
    for f in files:
        data.append(np.loadtxt(f))
    return data
if __name__== '__main__':
    parent_path = os.path.split(os.path.realpath(__file__))[0]
    data=[]    
    data.append(select(construct('U2.1_2016-1-7_16-8-13.txt')))
    data.append(select(construct('U2.1_2016-1-7_16-9-16.txt')))
    data.append(select(construct('U2.2_2016-1-7_16-10-57.txt')))
    data.append(select(construct('U2.1_2016-1-7_16-27-17.txt')))
    data.append(select(construct('U1.9_2016-1-15_17-35-29.txt')))
    data.append(select(construct('U1.9_2016-1-15_17-36-18.txt')))
    data.append(select(construct('U2_2016-1-16_10-34-23.txt')))
    data.append(select(construct('U2.0_2016-1-16_10-36-58.txt')))
    data.append(select(construct('U1.9_2016-1-16_11-0-42.txt')))
    data.append(select(construct('U1.9_2016-1-16_11-0-42.txt')))
    data.append(loadgz('../dataset/multispeed_dataset160118.pkl.gz'))
    data.append(loadgz('../dataset/dataset160118.pkl.gz'))
    visualize(data[-1])
    
    combined=combine(data)
    test=combined[6000:16000]
    train=np.vstack((combined[0:6000],combined[16000:70000]))
    del combined
    map(len,data)
    
    savefile(data,parent_path+'/dataset/TotalDataset.pkl.gz')
    savefile(train,parent_path+'/dataset/trainset.pkl.gz')
    savefile(test,parent_path+'/dataset/testset.pkl.gz')

        
    