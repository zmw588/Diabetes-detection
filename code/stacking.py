# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:32:46 2021

@author: 417-02
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import precision_score, roc_curve, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier 
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from functools import reduce 
from sklearn.tree import DecisionTreeClassifier
## 导入数据集切割训练与测试数据

data_train=np.load('K:/diabetes/X_train_diabetes_wavelet.npy')
label_train=np.load('K:/diabetes/Y_train_diabetes.npy')
data_test=np.load('K:/diabetes/X_test_diabetes_wavelet.npy')
label_test=np.load('K:/diabetes/Y_test_diabetes.npy')

print(data_train.shape)
print(data_test.shape)
num = str(label_train.tolist()).count("1")
num1 = str(label_train.tolist()).count("0")
print(num)
print(num1)

num = str(label_test.tolist()).count("1")
num1 = str(label_test.tolist()).count("0")
print(num)
print(num1)

from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=0)
oversample = SMOTETomek(random_state=0)
data_train, label_train = oversample.fit_resample(data_train,label_train)
data_test, label_test = oversample.fit_resample(data_test,label_test)

print(data_train.shape)
print(data_test.shape)
num = str(label_train.tolist()).count("1")
num1 = str(label_train.tolist()).count("0")
print(num)
print(num1)
num = str(label_test.tolist()).count("1")
num1 = str(label_test.tolist()).count("0")
print(num)
print(num1)
def SelectModel(modelname):
 
    if modelname == "SVM":
        
        model = SVC(kernel='rbf', C=16, gamma=0.125,probability=True)
 
 
    elif modelname == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=30)
 
    
    elif modelname == "ET":
        model = ExtraTreesClassifier(random_state=30)
        
    elif modelname == "XGBOOST":
        #from xgboost import XGBClassifier
        model = XGBClassifier(random_state=30)
 
    else:
        pass
    return model
 
def get_oof(clf,n_folds,X_train,y_train,X_test):
    ntrain = X_train.shape[0]
    ntest =  X_test.shape[0]
    classnum = len(np.unique(y_train))
    kf = KFold(n_splits=n_folds,shuffle=True, random_state = 3)
    oof_train = np.zeros((ntrain,classnum))
    oof_test = np.zeros((ntest,classnum))
 
 
    for i,(train_index, test_index) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_index] # 数据
        kf_y_train = y_train[train_index] # 标签
 
        kf_X_test = X_train[test_index]  # k-fold的验证集
 
        clf.fit(kf_X_train, kf_y_train)
        oof_train[test_index] = clf.predict_proba(kf_X_test)
 
        oof_test += clf.predict_proba(X_test)
    oof_test = oof_test/float(n_folds)
    return oof_train, oof_test


def draw_roc_curve( test_pre_proba,  test_auc, model_name):
    #fpr, tpr, roc_auc = train_pre_proba
    test_fpr, test_tpr, test_roc_auc = test_pre_proba

    plt.figure()
    lw = 2
    #plt.plot(fpr, tpr, color='darkorange',
    #         lw=lw, label='ROC curve (area = %0.2f)' % train_auc)
    plt.plot(test_fpr, test_tpr, color='red',
             lw=lw, label='ROC curve (area = %0.2f)' % test_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name)
    plt.legend(loc="lower right")
    plt.show()

LR = LogisticRegression()
LR.fit(data_train,label_train)

svc = SVC(kernel='rbf', C=16, gamma=0.125,probability=True)
svc.fit(data_train,label_train)

DT = DecisionTreeClassifier(max_depth=6)
DT.fit(data_train,label_train)

RF = RandomForestClassifier(random_state=30)
RF.fit(data_train,label_train)

GBDT=GradientBoostingClassifier()

XGB = XGBClassifier()
XGB.fit(data_train,label_train)

KNN=knn()
KNN.fit(data_train,label_train)

ET = ExtraTreesClassifier(random_state=30)
ET.fit(data_train,label_train)

models = [svc,RF,XGB,ET]
names = ["SVM", "RF","Xgb",'ET']
for name, model in zip(names, models):
    y_train_pred = model.predict_proba(data_train)[:, 1]
    y_test_pred = model.predict_proba(data_test)[:, 1]

    #train_roc = roc_curve(label_train, y_train_pred)
    test_roc = roc_curve(label_test, y_test_pred)

    #train_auc = roc_auc_score(label_train, y_train_pred)
    test_auc = roc_auc_score(label_test, y_test_pred)

    draw_roc_curve( test_roc, test_auc, name) 
# 单纯使用一个分类器的时候
clf_second = RandomForestClassifier(random_state=30)
clf_second.fit(data_train, label_train)
pred = clf_second.predict(data_test)
accuracy = metrics.accuracy_score(label_test, pred)*100
print('\n')
print (accuracy)
print(accuracy_score(pred, label_test))
print(classification_report(label_test, pred))


clf_second1 = XGBClassifier()
clf_second1.fit(data_train, label_train)
pred = clf_second1.predict(data_test)
accuracy = metrics.accuracy_score(label_test, pred)*100
print('\n')
print (accuracy)
print(accuracy_score(pred, label_test))
print(classification_report(label_test, pred))

clf_second2 = SVC(kernel='rbf', C=16, gamma=0.125,probability=True)
clf_second2.fit(data_train, label_train)
pred = clf_second2.predict(data_test)
accuracy = metrics.accuracy_score(label_test, pred)*100
print('\n')
print (accuracy)
print(accuracy_score(pred, label_test))
print(classification_report(label_test, pred))
 
clf_second3=ExtraTreesClassifier(random_state=30)
#clf_second3 = SVC(kernel='rbf', C=16, gamma=0.125,probability=True)
clf_second3.fit(data_train, label_train)
pred = clf_second3.predict(data_test)
accuracy = metrics.accuracy_score(label_test, pred)*100
print('\n')
print (accuracy)
print(accuracy_score(pred, label_test))
print(classification_report(label_test, pred))
# 使用stacking方法的时候
# 第一级，重构特征当做第二级的训练集
modelist = ['ET','XGBOOST','SVM','RF']
newfeature_list = []
newtestdata_list = []
for modelname in modelist:
    clf_first = SelectModel(modelname)
    oof_train_ ,oof_test_= get_oof(clf=clf_first,n_folds=10,X_train=data_train,y_train=label_train,X_test=data_test)
    newfeature_list.append(oof_train_)
    newtestdata_list.append(oof_test_)
 
# 特征组合
newfeature = reduce(lambda x,y:np.concatenate((x,y),axis=1),newfeature_list)    
newtestdata = reduce(lambda x,y:np.concatenate((x,y),axis=1),newtestdata_list)
 
 
# 第二级，使用上一级输出的当做训练集
clf_second1 = knn()

clf_second1.fit(newfeature, label_train)
pred = clf_second1.predict(newtestdata)
accuracy = metrics.accuracy_score(label_test, pred)*100
print('\n')
print (accuracy)
print(accuracy_score(pred, label_test))
print(classification_report(label_test, pred))

clf_second1.fit(newfeature, label_train)

y_test_pred = clf_second1.predict_proba(newtestdata)[:, 1]

test_roc = roc_curve(label_test, y_test_pred)

test_auc = roc_auc_score(label_test, y_test_pred)
name='stacking'
draw_roc_curve( test_roc, test_auc, name) 