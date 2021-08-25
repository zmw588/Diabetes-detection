# -*- coding: utf-8 -*-


import os
import numpy as np
import scipy.signal as signal

RATIO = 0.2
txt_path = 'H:/data/'
save_path = 'H:/diabetes/'
files = os.listdir(txt_path)


def getDataSet(number, X_data, Y_data):
 
    print("正在读取 " + str(number) + " 号波形数据...")
    x1 = np.loadtxt('H:/data/' + number)
    b, a = signal.butter(8, 0.02, 'lowpass')#配置滤波器 8 表示滤波器的阶数
    x2 = signal.filtfilt(b, a, x1)
    m=x2.tolist()
    data0 = [(x - min(m))/(max(m) - min(m)) for x in m]
    x=np.array(data0)

    Rclass1 = number[0]
    
    y=x2[signal.argrelextrema(x2, np.greater)]
    y1=x2[signal.argrelextrema(-x2, np.greater)]

    a=[]
    for i in signal.argrelextrema(x2,np.greater)[0]:

      if x2[i]>(np.amax(max(y))*0.95):
        a.append(i)
      print(a)   
     
    for i in a:
        i11=i+5
        i21=i-5
        i1=i+10
        i2=i-10
        i3=i+20
        i4=i-20
        i5=i+15
        i6=i-15
        a=np.append(a,i11)
        a=np.append(a,i21)
        a=np.append(a,i1)
        a=np.append(a,i2)
        a=np.append(a,i4)
        a=np.append(a,i3)
        a=np.append(a,i5)
        a=np.append(a,i6)
    print(a)    

    numberSet=len(a)
    print(numberSet)
    print(Rclass1)

    for i in range(numberSet):
       lable = Rclass1
       x_train = x[int(a[i]) - 160:int(a[i]) + 450]       
       if len(x_train)==610: 
          X_data.append(x_train)
          Y_data.append(lable)
          print(len(X_data))
    return

    

# 加载数据集并进行预处理
def loadData():
    dataSet = []
    lableSet = []
    for file in files: #遍历文件夹
         position = txt_path+'\\'+ file
         file_name = os.path.basename(position)

         getDataSet(file_name, dataSet, lableSet)

    print(np.array(lableSet).shape)    
    dataSet = np.array(dataSet).reshape(-1, 610)
    lableSet = np.array(lableSet).reshape(-1, 1)
    print(lableSet.shape)
    print(dataSet.shape)
    train_ds = np.hstack((dataSet, lableSet))
    print(train_ds.shape)
    np.random.shuffle(train_ds)

    # 数据集及其标签集
    X = train_ds[:, :610]
    Y = train_ds[:, 610]
    

    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index))
    print(test_length)
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    Y_train=Y_train.reshape(-1,1)
    Y_test=Y_test.reshape(-1,1)
    print(X_test.shape)
   
    print(Y_test.shape)
    
    d=[]
    d1=[]
    x1=[]
    d2=[]
    d3=[]
    for i in Y_train:
        i1=map(eval,i)
        i2=list(i1)        
        d.append(i2)
    Y_train=np.array(d)    

    for i in Y_test:
        i1=map(eval,i)
        i2=list(i1)
        d1.append(i2)
    Y_test=np.array(d1) 
    print(Y_test[0])
    
    for i in X_train:      
        i1=map(eval,i)
        i2=list(i1)            
        d2.append(i2)
        
    X_train=np.array(d2)  
    print(X_train[0])
    for i in X_test:        
        i1=map(eval,i)
        i2=list(i1)
        d3.append(i2)
        
    X_test=np.array(d3)  
    print(X_train.shape)
    print(X_test.shape)
    
    return X_train, Y_train, X_test, Y_test

def main():

    X_train, Y_train, X_test, Y_test = loadData()
    np.save("H:/diabetes/X_train_diabetes.npy",X_train)
    np.save("H:/diabetes/X_test_diabetes.npy",X_test)
    np.save("H:/diabetes/Y_train_diabetes.npy",Y_train)
    np.save("H:/diabetes/Y_test_diabetes.npy",Y_test)

if __name__ == '__main__':
    main()
