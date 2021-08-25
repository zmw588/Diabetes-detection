# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:09:53 2021

@author: emma
"""

import os
import datetime

import wfdb
import pywt
import seaborn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
d=np.zeros((48,), dtype=float)
a=np.load("K:/diabetes/X_train_diabetes.npy")
print(a.shape)
print(len(a))
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db6', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    a=[]
    a=np.hstack((coeffs[2],coeffs[3],coeffs[4]))
    return a
for i in range(len(a)):
    x2=denoise(a[i])
    print(x2.shape)
    d = np.vstack((d, x2))
d=d[1:,:]
print(d.shape)    
np.save("K:/diabetes/X_train_diabetes_wavelet.npy",d)

