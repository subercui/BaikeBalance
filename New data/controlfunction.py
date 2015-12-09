# -*- coding: utf-8 -*-
'''
实践何家瑞实验室原有控制算法，检验数据是否完全正确
author: CUI HAOTIAN
date:2015.12.05
'''
import numpy as np
import cPickle,gzip,os 

r2d=180./np.pi
k1=-4.5
k2=-0.5
k3=2.87

#death zone nonlinear function
def nonlinear(data,thrs):
    d_sign=np.sign(data)
    d_abs=np.abs(data)
    temp=np.fmax((d_abs-thrs),0)
    result=d_sign*temp
    return result

#给inputs vector输出output，接口同model.predict
def control_func(inputs,batch_size):
    '''inputs should includes
    (lean angle/degree, lean angle rate/degree, U0, turn angle in/rad, velocity)
    '''
    lean_angle=inputs[:,0]/r2d
    lean_angle_rate=inputs[:,1]/r2d
    U0=inputs[:,2]
    turn_angle_in=inputs[:,3]
    
    #1.death angle 0.0065
    temp1=nonlinear((-turn_angle_in-lean_angle),0.0065)
    #2.death_rate 0.015
    temp2=nonlinear((-lean_angle_rate),0.015)
    
    result=k1*temp1+k2*temp2+k3*turn_angle_in
    result=result*r2d
    
    return result
    
def absolute_error(predictions, targets):
    return np.abs(predictions - targets).mean(axis=0)
    
def mean_square_error(predictions, targets):
    return np.square(predictions - targets).mean(axis=0)
    
def load():
    parent_path = os.path.split(os.path.realpath(__file__))[0]
    f = gzip.open(parent_path+'/dataset/dataset.pkl.gz', 'rb')   
    data = cPickle.load(f)
    f.close()
    return data
    
if __name__=='__main__':
    array=np.array([[ -0.08, -0.0002, 1.9, 0,  2]])
    array=np.array([[ -0.006, -0.0002, 1.9, 0,  2]])
    result=control_func(array,array.shape[0])
    print '0位输出检测',result
    
    data=load()
    predictions=control_func(data[:,:5],data.shape[0])
    targets=data[:,-1]
    print 'absolute error ',absolute_error(predictions, targets)
    print 'mean square error ',mean_square_error(predictions, targets)
    
'''
now:    
absolute error  0.883747489747
mean square error  1.24495465467

after NN even better seen this:
0.45410999288624504
0.56617206440243195
'''

    
    
