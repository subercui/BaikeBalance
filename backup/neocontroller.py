# -*- coding: utf-8 -*-
import theano, theano.tensor as T
import numpy as np
from matplotlib import pyplot as plt
#import theano_lstm
import random,re,os,time,serial,math
import cPickle, gzip
import math
from modelfunction import *

pi=math.pi


def init(length,n_hiddens):
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    MLPmodel=build_mlp(5,1, 100,20)
    MLPmodel.load_weights('MLP_weightsMultispeed151226.hdf5')
   # MLPmodel.load_weights('MLP_weightsBest.hdf5')
    
    return MLPmodel
    

def hexShow(argv):  
    result = ''  
    hLen = len(argv)  
    for i in xrange(hLen):  
        hvol = ord(argv[i])  
        hhex = '%02x'%hvol  
        result += hhex  
    print 'hexShow:',result
    return result
    
def clipaste(a,last):
    a=hexShow(a)
    '''
    3 conditions:
    no ff;ff after ff>=54;ff after ff<54;
    3 measures:
    1) if (last,all)<54:newlast=(last,all),a='' else a=(last,all)
    2) a=latter54, newlast=after(latter+54)
    3) a=(last,former),newlast=latter
    '''
    idx=a.find('ff0')
    if idx<0:
        a=last+a
        newlast=''
        if len(a)<54:
            newlast=a
            a=''
    else:
        if len(a)-idx>=54:
            newlast=a[idx+54:]
            a=a[idx:idx+54]
        else:
            newlast=a[idx:]
            a=last+a[:idx]
    return a,newlast

def genOutput(a,RNNobj):
    if len(a)>=44:
        content=[]
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
        print '输入目标角度',turncontrol/10000.-1.
        #f.write('输入目标角度'+str(turncontrol/10000.-1.)+'\n')
        
        speed=speed/1000.
        if speed > 6:
            speed=0.
        print '输入倾斜角',lean/10000.-1.
        #f.write('输入倾斜角'+str(lean/10000.-1.)+'\n')
        lean=(lean/10000.-1.)*180/pi
        leanspeed=leanspeed/10000.-1
        U0=U0/1000.
        turncontrol=(turncontrol/10000.-1.)#rad
        content.append(str(time.time()))
        content.append(str(speed))
        content.append(str(lean))
        content.append(str(leanspeed))
        content.append(str(U0))
        print '输入U0',U0
        content.append(str(turncontrol))
        inputs=np.array([[lean,leanspeed,U0,turncontrol,speed]], dtype='float32')
        print inputs
        #inputs scale
        inputs = inputs-np.array([[0.,0.,1.9,0.,2.]])
        inputs = inputs/np.array([[7.7,10.0,1,0.03,0.5]])
        #predict
        result=RNNobj.predict(inputs,1)
        turnangle=result[0,0]
        #outputs scale and encode
        #turnangle=result[0][0,0,0]*(para_max[0,0,5]-para_min[0,0,5])+para_min[0,0,5]
        #输出手动非线性
        #turnangle=turnangle+2.75
        #turnangle=np.sign(turnangle)*6*np.sqrt(np.abs(turnangle))
        if turnangle>0:
            turnangle=min(turnangle,40.)
        else:
            turnangle=max(turnangle,-40.)
        print '输出角度',turnangle
        content.append(str(turnangle))
        #f.write('输出角度'+str(turnangle/180*pi)+'\n')
        turnangle=int((turnangle/180.*pi+1)*10000)
        #brake=np.argmax(result[1])
        brake=2
        if brake>=2:
            U=int(U0*1000)
            brake=75
        elif brake==1:
            U=int(1.5*1000)
            brake=68
        else:
            U=int(1.5*1000)
            brake=62
        content.append(str(U/1000.))
        content.append(str(brake))
        U="%04x"%U
        turnangle="%04x"%turnangle
        brake="%02x"%brake
        tosend='ff000702'+U+turnangle+brake
        #^ check
        check=int(tosend[0:2],16)
        for i in xrange(2,18,2):
            check=check^int(tosend[i:i+2],16)
        tosend=tosend+"%02x"%check
        genoutput(content)
    else:
        tosend=''
    
    return tosend

def genoutput(content):
    outputline=''
    for i in xrange(len(content)):
        outputline=outputline+content[i]+' '
    outputline=outputline+'\n'
    f.write(outputline)

def main():
    length=1
    n_hiddens=[100]
    RNNobj=init(length,n_hiddens)
    ser = serial.Serial('/dev/ttyS0',115200,timeout=0)
    last=''
    while(1):
        now=time.time()
        a=ser.readall()
        a,last=clipaste(a,last) 
        tosend=genOutput(a,RNNobj)
        hexsend=tosend.decode('hex')
        ser.write(hexsend)
        print 1000*(time.time()-now),'ms'
        while time.time()-now<0.00999:
            continue
        print 1000*(time.time()-now),'ms'
    ser.close()
    
if __name__== '__main__':
    f=open('log'+datetime.today().strftime("%y%m%d%H%M"),'wb')
    main()
    f.close()
