import serial,time

def hexShow(argv):  
    result = ''  
    hLen = len(argv)  
    for i in xrange(hLen):  
        hvol = ord(argv[i])  
        hhex = '%02x'%hvol  
        result += hhex  
    #print 'hexShow:',result
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
        

ser=serial.Serial('/dev/ttyS0',115200,timeout=0)
last=''
while(1):
    now=time.time()
    #a=ser.read(27)
    a=ser.readall()
    print 'a:',a,'\nhex a:',hexShow(a)
    a,last=clipaste(a,last)
    print 'a:',a,'last:',last
    hexsend=a.decode('hex')
    ser.write(hexsend)
    print 1000*(time.time()-now),'ms'
    while time.time()-now<0.00999:
        continue
    print 1000*(time.time()-now),'ms'
ser.close()

