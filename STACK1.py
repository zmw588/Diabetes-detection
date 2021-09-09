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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from functools import reduce 
from sklearn.tree import DecisionTreeClassifier
## 导入数据集切割训练与测试数据
np.random.seed(10)


X_train1=np.load('D:/417_students/zhangMW/ppg1-data/X_train_diabetes_wavelet1.npy')
X_train2=np.load('D:/417_students/zhangMW/ppg1-data/X_train_diabetes_wavelet2.npy')
Y_train1=np.load('D:/417_students/zhangMW/ppg1-data/Y_train_diabetes1.npy')
Y_train2=np.load('D:/417_students/zhangMW/ppg1-data/Y_train_diabetes2.npy')
data_test1=np.load('D:/417_students/zhangMW/ppg1-data/X_test_diabetes_wavelet1.npy')
data_test2=np.load('D:/417_students/zhangMW/ppg1-data/X_test_diabetes_wavelet2.npy')
label_test1=np.load('D:/417_students/zhangMW/ppg1-data/Y_test_diabetes1.npy')
label_test2=np.load('D:/417_students/zhangMW/ppg1-data/Y_test_diabetes2.npy')
#X_test, Y_test = data_test1, label_test1
#X_train, Y_train = data_train1, label_train1

data_train2=np.load('D:/417_students/zhangMW/ppg1-data/X_train_diabetes_wavelet.npy')
label_train2=np.load('D:/417_students/zhangMW/ppg1-data/Y_train_diabetes.npy')
data_test0=np.load('D:/417_students/zhangMW/ppg1-data/X_test_diabetes_wavelet.npy')
label_test0=np.load('D:/417_students/zhangMW/ppg1-data/Y_test_diabetes.npy')

X_train=np.vstack((X_train1,X_train2))
Y_train=np.vstack((Y_train1,Y_train2))


data_train=np.vstack((X_train,data_train2))
label_train=np.vstack((Y_train,label_train2))


data_test=data_test0
label_test=label_test0    

print(data_train.shape)
print(data_test.shape)
num = str(label_train.tolist()).count("1")
num1 = str(label_train.tolist()).count("0")
print(num)  
print(num1)


from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=0)
oversample = SMOTETomek(random_state=0)
data_train, label_train = oversample.fit_resample(data_train,label_train)
#data_test, label_test = oversample.fit_resample(data_test,label_test)

print(data_train.shape)
print(data_test.shape)
num = str(label_train.tolist()).count("1")
num1 = str(label_train.tolist()).count("0")
print(num)
print(num1)

def SelectModel(modelname):
 
    if modelname == "SVM":
        
        model = SVC(kernel='rbf', C=0.25,probability=True)
 
 
    elif modelname == "RF":
#        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=30)
 
    
    elif modelname == "ET":
        model = ExtraTreesClassifier(random_state=30)
        
    elif modelname == "XGBOOST":
        #from xgboost import XGBClassifier
        model = XGBClassifier(random_state=30)
 
    else:
        pass
    return model
 
def get_oof(clf,n_folds,X_train,y_train,X_test,X_test1,X_test2):
    ntrain = X_train.shape[0]
    ntest =  X_test.shape[0]
    ntest1 =  X_test1.shape[0]
    ntest2 =  X_test2.shape[0]
    classnum = len(np.unique(y_train))
    kf = KFold(n_splits=n_folds,shuffle=True, random_state = 3)
    oof_train = np.zeros((ntrain,classnum))
    oof_test = np.zeros((ntest,classnum))
    oof_test1 = np.zeros((ntest1,classnum))
    oof_test2 = np.zeros((ntest2,classnum))
 
    for i,(train_index, test_index) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_index] # 数据
        kf_y_train = y_train[train_index] # 标签
 
        kf_X_test = X_train[test_index]  # k-fold的验证集
 
        clf.fit(kf_X_train, kf_y_train)
        oof_train[test_index] = clf.predict_proba(kf_X_test)
 
        oof_test += clf.predict_proba(X_test)
        oof_test1 += clf.predict_proba(X_test1)
        oof_test2 += clf.predict_proba(X_test2)
    oof_test = oof_test/float(n_folds)
    oof_test1 = oof_test1/float(n_folds)
    oof_test2 = oof_test2/float(n_folds)
    return oof_train, oof_test,oof_test1,oof_test2


def draw_roc_curve( test_pre_proba, test_auc, test_pre_proba1, test_auc1,test_pre_proba2, test_auc2,model_name):
    #fpr, tpr, roc_auc = train_pre_proba
    test_fpr, test_tpr, test_roc_auc = test_pre_proba
    test_fpr1, test_tpr1, test_roc_auc1 = test_pre_proba1
    test_fpr2, test_tpr2, test_roc_auc2 = test_pre_proba2

      
    plt.figure(dpi = 400)
    plt.plot(test_fpr, test_tpr, lw=2, label = 'Dataset1 = %0.3f' % test_auc, color='Red')
    plt.plot(test_fpr1, test_tpr1,  label = 'Dataset2 = %0.3f' % test_auc1, color='k')
    plt.plot(test_fpr2, test_tpr2,  label = 'Dataset3 = %0.3f' % test_auc2, color='RoyalBlue')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
#    plt.tick_params(labelsize=15)
    plt.title(model_name)
    plt.show()

# 使用stacking方法的时候
# 第一级，重构特征当做第二级的训练集
modelist = ['ET','XGBOOST','SVM','RF']
newfeature_list = []
newtestdata_list = []
newfeature_list1 = []
newtestdata_list1 = []
newfeature_list2 = []
newtestdata_list2 = []

for modelname in modelist:
    clf = SelectModel(modelname)
    clf.fit(data_train, label_train)
    pred = clf.predict(data_test)
    accuracy = metrics.accuracy_score(label_test, pred)*100
    print('\n')
    print(modelname)
    print (accuracy)
    print(accuracy_score(pred, label_test))
    print(classification_report(label_test, pred))
    y_test_pred = clf.predict_proba(data_test)[:, 1]
    test_roc = roc_curve(label_test, y_test_pred)
    test_auc = roc_auc_score(label_test, y_test_pred)
    
    pred1 = clf.predict(data_test1)
    accuracy1 = metrics.accuracy_score(label_test1, pred1)*100
    print('\n')
    print (accuracy1)
    print(accuracy_score(pred1, label_test1))
    print(classification_report(label_test1, pred1))
    y_test_pred1 = clf.predict_proba(data_test1)[:, 1]
    test_roc1 = roc_curve(label_test1, y_test_pred1)
    test_auc1 = roc_auc_score(label_test1, y_test_pred1)
    
    pred2 = clf.predict(data_test2)
    accuracy2 = metrics.accuracy_score(label_test2, pred2)*100
    print('\n')
    print (accuracy2)
    print(accuracy_score(pred2, label_test2))
    print(classification_report(label_test2, pred2))
    y_test_pred2 = clf.predict_proba(data_test2)[:, 1]
    test_roc2 = roc_curve(label_test2, y_test_pred2)
    test_auc2 = roc_auc_score(label_test2, y_test_pred2)
    
    draw_roc_curve( test_roc, test_auc, test_roc1, test_auc1,test_roc2, test_auc2,modelname) 
    
    
for modelname in modelist:
    clf_first = SelectModel(modelname)
    oof_train_ ,oof_test_,oof_test_1,oof_test_2= get_oof(clf=clf_first,n_folds=10,X_train=data_train,y_train=label_train,X_test=data_test,X_test1=data_test1,X_test2=data_test2)
    newfeature_list.append(oof_train_)
    newtestdata_list.append(oof_test_)
    newtestdata_list1.append(oof_test_1)
    newtestdata_list2.append(oof_test_2) 
# 特征组合
newfeature = reduce(lambda x,y:np.concatenate((x,y),axis=1),newfeature_list)    
newtestdata = reduce(lambda x,y:np.concatenate((x,y),axis=1),newtestdata_list)
 
newtestdata1 = reduce(lambda x,y:np.concatenate((x,y),axis=1),newtestdata_list1)
 
newtestdata2 = reduce(lambda x,y:np.concatenate((x,y),axis=1),newtestdata_list2) 
 
# 第二级，使用上一级输出的当做训练集
clf_second1 = knn()

clf_second1.fit(newfeature, label_train)
pred = clf_second1.predict(newtestdata)
accuracy = metrics.accuracy_score(label_test, pred)*100
print('\n')
print('stacking')
print (accuracy)
print(accuracy_score(pred, label_test))
print(classification_report(label_test, pred))

pred1 = clf_second1.predict(newtestdata1)
accuracy1 = metrics.accuracy_score(label_test1, pred1)*100
print('\n')
print (accuracy1)
print(accuracy_score(pred1, label_test1))
print(classification_report(label_test1, pred1))

pred2 = clf_second1.predict(newtestdata2)
accuracy2 = metrics.accuracy_score(label_test2, pred2)*100
print('\n')
print (accuracy2)
print(accuracy_score(pred2, label_test2))
print(classification_report(label_test2, pred2))


y_test_pred = clf_second1.predict_proba(newtestdata)[:, 1]
test_roc = roc_curve(label_test, y_test_pred)
test_auc = roc_auc_score(label_test, y_test_pred)

y_test_pred1 = clf_second1.predict_proba(newtestdata1)[:, 1]
test_roc1 = roc_curve(label_test1, y_test_pred1)
test_auc1 = roc_auc_score(label_test1, y_test_pred1)

y_test_pred2 = clf_second1.predict_proba(newtestdata2)[:, 1]
test_roc2 = roc_curve(label_test2, y_test_pred2)
test_auc2 = roc_auc_score(label_test2, y_test_pred2)

name='Stacking'
draw_roc_curve( test_roc, test_auc,test_roc1, test_auc1,test_roc2, test_auc2, name) 