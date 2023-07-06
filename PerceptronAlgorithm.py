#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np 
import pandas as pd

train = pd.read_csv('train.data', header=None)

def PerceptronTrain(X_train, y_train, MaxIter=20):
    w_ = np.zeros(X_train.shape[1])
    b=0
    for _ in range(MaxIter):
        for xi,yi in zip(X_train, y_train):
            a = (np.dot(xi, w_)+b) 
            if yi*a <= 0:
                w_ += yi*xi
                b += yi

    return w_,b

#get predicted labels
def PerceptronTest(b, w_, X):
    a = np.dot(X, w_) + b
    return np.sign(a)

def accuracy(y_train, y_true):
    num_correct=0.0
    predicts = []
    for i in range(len(y_train)):
        predict = y_train
        predicts.append(predict)
                          
        if predict[i]==y_true[i]:
            num_correct+=1.0
    return num_correct/float(len(y_train))

def PerceptronTrainReg(X_train, y_train, MaxIter=20):
    w_ = np.zeros(X_train.shape[1]) 
    b=0
    lamb_ = 0.01

    
    for _ in range(MaxIter):
        for xi,yi in zip(X_train, y_train):
            a = (np.dot(xi, w_)+b) 
            if yi*a <= 0:
                w_ +=yi*xi
                W_ = (1-2*lamb_)*w_
                b += yi
            else:
                w_ = w_
                W_ = w_*(1-2*lamb_)
                b = b
    return W_,b

def PerceptronTrainReg2(X_train, y_train, MaxIter=20):
    w_ = np.zeros(X_train.shape[1]) 
    b=0
    lamb_ = 0.1

    
    for _ in range(MaxIter):
        for xi,yi in zip(X_train, y_train):
            a = (np.dot(xi, w_)+b) 
            if yi*a <= 0:
                w_ +=yi*xi
                W_ = (1-2*lamb_)*w_
                b += yi
            else:
                w_ = w_
                W_ = w_*(1-2*lamb_)
                b = b
    return W_,b

def PerceptronTrainReg3(X_train, y_train, MaxIter=20):
    w_ = np.zeros(X_train.shape[1]) 
    b=0
    lamb_ = 1.0

    
    for _ in range(MaxIter):
        for xi,yi in zip(X_train, y_train):
            a = (np.dot(xi, w_)+b) 
            if yi*a <= 0:
                w_ +=yi*xi
                W_ = (1-2*lamb_)*w_
                b += yi
            else:
                w_ = w_
                W_ = w_*(1-2*lamb_)
                b = b
    return W_,b

def PerceptronTrainReg4(X_train, y_train, MaxIter=20):
    w_ = np.zeros(X_train.shape[1]) 
    b=0
    lamb_ = 10.0

    
    for _ in range(MaxIter):
        for xi,yi in zip(X_train, y_train):
            a = (np.dot(xi, w_)+b) 
            if yi*a <= 0:
                w_ +=yi*xi
                W_ = (1-2*lamb_)*w_
                b += yi
            else:
                w_ = w_
                W_ = w_*(1-2*lamb_)
                b = b
    return W_,b

def PerceptronTrainReg5(X_train, y_train, MaxIter=20):
    w_ = np.zeros(X_train.shape[1]) 
    b=0
    lamb_ = 100.0

    
    for _ in range(MaxIter):
        for xi,yi in zip(X_train, y_train):
            a = (np.dot(xi, w_)+b) 
            if yi*a <= 0:
                w_ +=yi*xi
                W_ = (1-2*lamb_)*w_
                b += yi
            else:
                w_ = w_
                W_ = w_*(1-2*lamb_)
                b = b
    return W_,b


#categorising train dataframes
class1_df = train[train.loc[:,4] == 'class-1']
class2_df = train[train.loc[:,4] == 'class-2']
class3_df = train[train.loc[:,4] == 'class-3']

#training first binary class
data1 = pd.concat([class1_df, class2_df], ignore_index=True)
data1.rename(columns={4:'Class_1'}, inplace=True)
data1
dataset1 = pd.get_dummies(data1['Class_1'])
dataset1
X = pd.concat([data1, dataset1], axis=1)
X.drop(['Class_1'], axis=1, inplace=True)
X= X.replace(0, str(-1))
X = X.drop(['class-2'], axis=1)
X
X_train = X.loc[0:161,:3]
X_train = X_train.to_numpy()
X_train
y_train = X.loc[0:161,'class-1']
y_train = y_train.to_numpy()
y_train = y_train.astype(float)
y_train
training1 = PerceptronTrain(X_train, y_train)
Arr1 = training1[0][:4]

#training second binary class
data2 = pd.concat([class2_df, class3_df], ignore_index=True)
data2.rename(columns={4:'Class_2'}, inplace=True)
dataset2 = pd.get_dummies(data2['Class_2'])
dataset2
X2 = pd.concat([data2, dataset2], axis=1)
X2.drop(['Class_2'], axis=1, inplace=True)
X2= X2.replace(0, str(-1))
X2 = X2.drop(['class-3'], axis=1)
X2
X2_train = X2.loc[0:161,:3]
X2_train = X2_train.to_numpy()
X2_train
y2_train = X2.loc[0:161,'class-2']
y2_train = y2_train.to_numpy()
y2_train = y2_train.astype(float)
y2_train
training2 = PerceptronTrain(X2_train, y2_train)
Arr2 = training2[0][:4]

#training third binary class
data3 = pd.concat([class1_df, class3_df], ignore_index=True)
data3.rename(columns={4:'Class_3'}, inplace=True)
dataset3 = pd.get_dummies(data3['Class_3'])
dataset3
X3 = pd.concat([data3, dataset3], axis=1)
X3.drop(['Class_3'], axis=1, inplace=True)
X3= X3.replace(0, str(-1))
X3 = X3.drop(['class-3'], axis=1)
X3
X3_train = X3.loc[0:161,:3]
X3_train = X3_train.to_numpy()
X3_train
y3_train = X3.loc[0:161,'class-1']
y3_train = y3_train.to_numpy()
y3_train = y3_train.astype(float)
y3_train
training3 = PerceptronTrain(X3_train, y3_train)
Arr3 = training3[0][:4]

#get y predicted labels
PerceptronTest1 = PerceptronTest(-1.0, Arr1, X_train)
PerceptronTest2 = PerceptronTest(-5.0, Arr2, X2_train)
PerceptronTest3 = PerceptronTest(-2.0, Arr3, X3_train)


#get accuracy
Accuracy1 = accuracy(PerceptronTest1, y_train)
Accuracy2 = accuracy(PerceptronTest2, y2_train)
Accuracy3 = accuracy(PerceptronTest3, y3_train)


print(f"The Weighted Vector and Bias for Class 1 and Class 2 Train Dataset is: {training1}")
print(f"The Weighted Vector and Bias for Class 2 and Class 3 Train Dataset is: {training2}")
print(f"The Weighted Vector and Bias for Class 1 and Class 3 Train Dataset is: {training3}")
print(f"The accuracy score for Class 1 and Class 2 Train Dataset is: {Accuracy1}")
print(f"The accuracy score for Class 2 and Class 3 Train Dataset is: {Accuracy2}")
print(f"The accuracy score for Class 1 and Class 3 Train Dataset is: {Accuracy3}")

#Test Binary Class
test = pd.read_csv('test.data', header=None)

#categorising test dataframes
class1_dft = test[test.loc[:,4] == 'class-1']
class2_dft = test[test.loc[:,4] == 'class-2']
class3_dft = test[test.loc[:,4] == 'class-3']

#training first binary class
datat1 = pd.concat([class1_dft, class2_dft], ignore_index=True)
datat1.rename(columns={4:'Class_1'}, inplace=True)
dataset1t = pd.get_dummies(datat1['Class_1'])
dataset1t
Xt = pd.concat([datat1, dataset1t], axis=1)
Xt.drop(['Class_1'], axis=1, inplace=True)
Xt= Xt.replace(0, str(-1))
Xt = Xt.drop(['class-2'], axis=1)
Xt
Xt_train = Xt.loc[0:161,:3]
Xt_train = Xt_train.to_numpy()
Xt_train
y_traint = Xt.loc[0:161,'class-1']
y_traint = y_traint.to_numpy()
y_traint = y_traint.astype(float)
y_traint
training1t = PerceptronTrain(Xt_train, y_traint)
Arr1t = training1t[0][:4]


#training second binary class
datat2 = pd.concat([class2_dft, class3_dft], ignore_index=True)
datat2.rename(columns={4:'Class_2'}, inplace=True)
dataset2t = pd.get_dummies(datat2['Class_2'])
dataset2t
Xt2 = pd.concat([datat2, dataset2t], axis=1)
Xt2.drop(['Class_2'], axis=1, inplace=True)
Xt2= Xt2.replace(0, str(-1))
Xt2 = Xt2.drop(['class-3'], axis=1)
Xt2
Xt_train2 = Xt2.loc[0:161,:3]
Xt_train2 = Xt_train2.to_numpy()
Xt_train2
y_traint2 = Xt2.loc[0:161,'class-2']
y_traint2 = y_traint2.to_numpy()
y_traint2 = y_traint2.astype(float)
y_traint2
training2t = PerceptronTrain(Xt_train2, y_traint2)
Arr2t = training2t[0][:4]

#training third binary class
datat3 = pd.concat([class1_dft, class3_dft], ignore_index=True)
datat3.rename(columns={4:'Class_3'}, inplace=True)
dataset3t = pd.get_dummies(datat3['Class_3'])
dataset3t
Xt3 = pd.concat([datat3, dataset3t], axis=1)
Xt3.drop(['Class_3'], axis=1, inplace=True)
Xt3= Xt3.replace(0, str(-1))
Xt3 = Xt3.drop(['class-3'], axis=1)
Xt3
Xt_train3 = Xt3.loc[0:161,:3]
Xt_train3 = Xt_train3.to_numpy()
Xt_train3
y_traint3 = Xt3.loc[0:161,'class-1']
y_traint3 = y_traint3.to_numpy()
y_traint3 = y_traint3.astype(float)
y_traint3
training3t = PerceptronTrain(Xt_train3, y_traint3)
Arr3t = training3t[0][:4]


#get y predicted labels
PerceptronTest1t = PerceptronTest(0.0, Arr1t, Xt_train)
PerceptronTest2t = PerceptronTest(-2.0, Arr2t, Xt_train2)
PerceptronTest3t = PerceptronTest(1.0, Arr3t, Xt_train3)


#get accuracy
Accuracy1t = accuracy(PerceptronTest1t, y_traint)
Accuracy2t = accuracy(PerceptronTest2t, y_traint2)
Accuracy3t = accuracy(PerceptronTest3t, y_traint3)




print(f"The Weighted Vector and Bias for Class 1 and Class 2 Test Dataset is: {training1t}")
print(f"The Weighted Vector and Bias for Class 2 and Class 3 Test Dataset is: {training2t}")
print(f"The Weighted Vector and Bias for Class 1 and Class 3 Test Dataset is: {training3t}")
print(f"The accuracy score for Class 1 and Class 2 Test Dataset is: {Accuracy1t}")
print(f"The accuracy score for Class 2 and Class 3 Test Dataset is: {Accuracy2t}")
print(f"The accuracy score for Class 1 and Class 3 Test Dataset is: {Accuracy3t}")

#first classification
yv1_train = np.where(train.loc[:,4]=='class-1', 1, -1)
yv1_train
Xv1 = train
Xv1['yv1_train'] = yv1_train.tolist()
Xv1
X_trainer =  Xv1.loc[0:241, :3]
X_trainer = X_trainer.to_numpy()
X_trainer 
yv1_train = yv1_train.astype(float)
yv1_train
Model1 = PerceptronTrain(X_trainer, yv1_train)

#second classification 
yv2_train = np.where(train.loc[:,4]=='class-2', 1, -1)
yv2_train
Xv2 = train
Xv2['yv2_train'] = yv2_train.tolist()
Xv2
yv2_train = yv2_train.astype(float)
yv2_train
Model2 = PerceptronTrain(X_trainer, yv2_train)


#third classification 
yv3_train = np.where(train.loc[:,4]=='class-3', 1, -1)
yv3_train
Xv3 = train
Xv3['yv3_train'] = yv3_train.tolist()
Xv3
yv3_train = yv3_train.astype(float)
yv3_train
Model3 = PerceptronTrain(X_trainer, yv3_train)


#get predicted labels for each class
Mode1 = Model1[0][:4]
Mode2 = Model2[0][:4]
Mode3 = Model3[0][:4]

def PerceptronTest(b, w_, X):
    a = np.dot(X, w_) + b
    y = np.sign(a)
    return a, y

y_label1= PerceptronTest(-4.0, Mode1, X_trainer)
y_label2= PerceptronTest(-4.0, Mode2, X_trainer)
y_label3= PerceptronTest(-1.0, Mode3, X_trainer)

ylabel1 = y_label1
weight1 = ylabel1[0]
lab1 = ylabel1[1]
ylabel2 = y_label2
weight2 = ylabel2[0]
lab2 = ylabel2[1]
ylabel3 = y_label3
weight3 = ylabel3[0]
lab3 = ylabel3[1]


#concatenate for confident score
Xconc = np.vstack(([[weight1], [weight2], [weight3]]))
multiclassifier = np.argmax(Xconc, axis=0)
multiclassifier
multiclass = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifier]

def accuracy(y_train, y_true):
    num_correct=0.0
    predicts = []
    for i in range(len(y_train)):
        predict = y_train
        predicts.append(predict)
                          
        if predict[i]==y_true[i]:
            num_correct+=1.0
    return num_correct/float(len(y_train))

#train classifiers
trainclass = train.loc[0:241,4]
trainclass 

#Accuracy
Accuracym = accuracy(multiclass, trainclass)
Accuracym


print(f"The Weighted Vector and Bias for Class 1 Train Multiclass Dataset is: {Model1}")
print(f"The Weighted Vector and Bias for Class 2 Train Multiclass Dataset is: {Model2}")
print(f"The Weighted Vector and Bias for Class 3 Train Multiclass Dataset is: {Model3}")
print(f"The accuracy score for Multiclass Train is: {Accuracym}")

test = pd.read_csv('test.data', header=None)
#first classification
yv1_test = np.where(test.loc[:,4]=='class-1', 1, -1)
yv1_test
Xv1t = test
Xv1t['yv1_test'] = yv1_test.tolist()
Xv1t
X_tester =  Xv1t.loc[0:59, :3]
X_tester = X_tester.to_numpy()
X_tester
yv1_test = yv1_test.astype(float)
yv1_test
Model1t = PerceptronTrain(X_tester, yv1_test)


#second classification 
yv2_test = np.where(test.loc[:,4]=='class-2', 1, -1)
yv2_test
Xv2t = test
Xv2t['yv2_test'] = yv2_test.tolist()
Xv2t
yv2_test = yv2_test.astype(float)
yv2_test
Model2t = PerceptronTrain(X_tester, yv2_test)


#third classification 
yv3_test = np.where(test.loc[:,4]=='class-3', 1, -1)
yv3_test
Xv3t = test
Xv3t['yv3_test'] = yv3_test.tolist()
Xv3t
yv3_test = yv3_test.astype(float)
yv3_test
Model3t = PerceptronTrain(X_tester, yv3_test)

#get predicted labels for each class
Model1tt = Model1t[0][:4]
Model2tt = Model2t[0][:4]
Model3tt = Model3t[0][:4]

def PerceptronTest(b, w_, X):
    a = np.dot(X, w_) + b
    y = np.sign(a)
    return a, y

y_label1t= PerceptronTest(0.0, Model1tt, X_tester)
y_label2t= PerceptronTest(-2.0, Model2tt, X_tester)
y_label3t= PerceptronTest(-2.0, Model3tt, X_tester)

ylabel1t = y_label1t
weight1t = ylabel1t[0]
lab1t = ylabel1t[1]
ylabel2t = y_label2t
weight2t = ylabel2t[0]
lab2t = ylabel2t[1]
ylabel3t = y_label3t
weight3t = ylabel3t[0]
lab3t = ylabel3t[1]


#concatenate for confident score
Xconct = np.vstack(([[weight1t], [weight2t], [weight3t]]))
multiclassifiert = np.argmax(Xconct, axis=0)
multiclassifiert
multiclasst = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifiert]

def accuracy(y_train, y_true):
    num_correct=0.0
    predicts = []
    for i in range(len(y_train)):
        predict = y_train
        predicts.append(predict)
                          
        if predict[i]==y_true[i]:
            num_correct+=1.0
    return num_correct/float(len(y_train))

#test classifiers
dataframe1t = test.rename(columns={4:'Classes'}, inplace=True)
test = test.drop(['yv1_test'], axis=1)
test = test.drop(['yv2_test'], axis=1)
test = test.drop(['yv3_test'], axis=1)
y_testm = test.loc[0:59,'Classes']


#Accuracy
Accuracymt = accuracy(multiclasst, y_testm)
Accuracymt


print(f"The Weighted Vector and Bias for Class 1 Test Multiclass Dataset is: {Model1t}")
print(f"The Weighted Vector and Bias for Class 2 Test Multiclass Dataset is: {Model2t}")
print(f"The Weighted Vector and Bias for Class 3 Test Multiclass Dataset is: {Model3t}")
print(f"The accuracy score for Multiclass Test is: {Accuracymt}")

#Train dataset for Regularization
#Regularization with lambda 0.01
def PerceptronTrainReg(X_train, y_train, MaxIter=20):
    w_ = np.zeros(X_train.shape[1]) 
    b=0
    lamb_ = 0.01

    
    for _ in range(MaxIter):
        for xi,yi in zip(X_train, y_train):
            a = (np.dot(xi, w_)+b) 
            if yi*a <= 0:
                w_ +=yi*xi
                W_ = (1-2*lamb_)*w_
                b += yi
            else:
                w_ = w_
                W_ = w_*(1-2*lamb_)
                b = b
    return W_,b

#Regularization with lambda 0.01
Model1r = PerceptronTrainReg(X_trainer, yv1_train)
Model2r = PerceptronTrainReg(X_trainer, yv2_train)
Model3r = PerceptronTrainReg(X_trainer, yv3_train)

Mode1r = Model1r[0][:4]
Mode2r = Model2r[0][:4]
Mode3r = Model3r[0][:4]

#get Y labels
y_label1r= PerceptronTest(-4.0, Mode1r, X_trainer)
y_label2r= PerceptronTest(-4.0, Mode2r, X_trainer)
y_label3r= PerceptronTest(-1.0, Mode3r, X_trainer)

ylabel1r = y_label1r
weight1r = ylabel1r[0]
lab1r = ylabel1r[1]
ylabel2r = y_label2r
weight2r = ylabel2r[0]
lab2r = ylabel2r[1]
ylabel3r = y_label3r
weight3r = ylabel3r[0]
lab3r = ylabel3r[1]

#multiclassifier at 0.01 lambda L2 Regularisation
Xconcr = np.vstack(([[weight1r], [weight2r], [weight3r]]))
multiclassifierr = np.argmax(Xconcr, axis=0)
multiclassr = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifierr]

trainclassr = train.loc[0:241,4]
trainclassr
Accuracymr = accuracy(multiclassr, trainclassr)


print(f"The Weighted Vector and Bias for Class 1 Train Multiclass Dataset with L2 Regularization at lambda 0.01 is: {Model1r}")
print(f"The Weighted Vector and Bias for Class 2 Train Multiclass Dataset with L2 Regularization at lambda 0.01 is: {Model2r}")
print(f"The Weighted Vector and Bias for Class 3 Train Multiclass Dataset with L2 Regularization at lambda 0.01 is: {Model3r}")
print(f"The accuracy score for Multiclass Train with L2 Regularization at lambda 0.01 is: {Accuracymr}")

#Regularization with lambda 0.1
def PerceptronTrainReg2(X_train, y_train, MaxIter=20):
    w_ = np.zeros(X_train.shape[1]) 
    b=0
    lamb_ = 0.1

    
    for _ in range(MaxIter):
        for xi,yi in zip(X_train, y_train):
            a = (np.dot(xi, w_)+b) 
            if yi*a <= 0:
                w_ +=yi*xi
                W_ = (1-2*lamb_)*w_
                b += yi
            else:
                w_ = w_
                W_ = w_*(1-2*lamb_)
                b = b
    return W_,b


Model1r2 = PerceptronTrainReg2(X_trainer, yv1_train)
Model2r2 = PerceptronTrainReg2(X_trainer, yv2_train)
Model3r2 = PerceptronTrainReg2(X_trainer, yv3_train)

Mode1r2 = Model1r2[0][:4]
Mode2r2 = Model2r2[0][:4]
Mode3r2 = Model3r2[0][:4]

#get Y labels
y_label1r2= PerceptronTest(-4.0, Mode1r2, X_trainer)
y_label2r2= PerceptronTest(-4.0, Mode2r2, X_trainer)
y_label3r2= PerceptronTest(-1.0, Mode3r2, X_trainer)

ylabel1r2 = y_label1r2
weight1r2 = ylabel1r2[0]
lab1r2 = ylabel1r2[1]
ylabel2r2 = y_label2r2
weight2r2 = ylabel2r2[0]
lab2r2 = ylabel2r2[1]
ylabel3r2 = y_label3r2
weight3r2 = ylabel3r2[0]
lab3r2 = ylabel3r2[1]

#multiclassifier at 0.01 lambda L2 Regularisation
Xconcr2 = np.vstack(([[weight1r2], [weight2r2], [weight3r2]]))
multiclassifierr2 = np.argmax(Xconcr2, axis=0)
multiclassr2 = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifierr2]

Accuracym2 = accuracy(multiclassr2, trainclassr)


print(f"The Weighted Vector and Bias for Class 1 Train Multiclass Dataset with L2 Regularization at lambda 0.1 is: {Model1r2}")
print(f"The Weighted Vector and Bias for Class 2 Train Multiclass Dataset with L2 Regularization at lambda 0.1 is: {Model2r2}")
print(f"The Weighted Vector and Bias for Class 3 Train Multiclass Dataset with L2 Regularization at lambda 0.1 is: {Model3r2}")
print(f"The accuracy score for Multiclass Train with L2 Regularization at lambda 0.1 is: {Accuracym2}")

#Regularization with lambda 1.0
def PerceptronTrainReg3(X_train, y_train, MaxIter=20):
    w_ = np.zeros(X_train.shape[1]) 
    b=0
    lamb_ = 1.0

    
    for _ in range(MaxIter):
        for xi,yi in zip(X_train, y_train):
            a = (np.dot(xi, w_)+b) 
            if yi*a <= 0:
                w_ +=yi*xi
                W_ = (1-2*lamb_)*w_
                b += yi
            else:
                w_ = w_
                W_ = w_*(1-2*lamb_)
                b = b
    return W_,b


Model1r3 = PerceptronTrainReg3(X_trainer, yv1_train)
Model2r3 = PerceptronTrainReg3(X_trainer, yv2_train)
Model3r3 = PerceptronTrainReg3(X_trainer, yv3_train)

Mode1r3 = Model1r3[0][:4]
Mode2r3 = Model2r3[0][:4]
Mode3r3 = Model3r3[0][:4]

#get Y labels
y_label1r3= PerceptronTest(-4.0, Mode1r3, X_trainer)
y_label2r3= PerceptronTest(-4.0, Mode2r3, X_trainer)
y_label3r3= PerceptronTest(-1.0, Mode3r3, X_trainer)

ylabel1r3 = y_label1r3
weight1r3 = ylabel1r3[0]
lab1r3 = ylabel1r3[1]
ylabel2r3 = y_label2r3
weight2r3 = ylabel2r3[0]
lab2r3 = ylabel2r3[1]
ylabel3r3 = y_label3r3
weight3r3 = ylabel3r3[0]
lab3r3 = ylabel3r3[1]

#multiclassifier at 0.01 lambda L2 Regularisation
Xconcr3 = np.vstack(([[weight1r3], [weight2r3], [weight3r3]]))
multiclassifierr3 = np.argmax(Xconcr3, axis=0)
multiclassr3 = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifierr3]

Accuracym3 = accuracy(multiclassr3, trainclassr)


print(f"The Weighted Vector and Bias for Class 1 Train Multiclass Dataset with L2 Regularization at lambda 1.0 is: {Model1r3}")
print(f"The Weighted Vector and Bias for Class 2 Train Multiclass Dataset with L2 Regularization at lambda 1.0 is: {Model2r3}")
print(f"The Weighted Vector and Bias for Class 3 Train Multiclass Dataset with L2 Regularization at lambda 1.0 is: {Model3r3}")
print(f"The accuracy score for Multiclass Train with L2 Regularization at lambda 1.0 is: {Accuracym3}")

#Regularization with lambda 10.0
def PerceptronTrainReg4(X_train, y_train, MaxIter=20):
    w_ = np.zeros(X_train.shape[1]) 
    b=0
    lamb_ = 10.0

    
    for _ in range(MaxIter):
        for xi,yi in zip(X_train, y_train):
            a = (np.dot(xi, w_)+b) 
            if yi*a <= 0:
                w_ +=yi*xi
                W_ = (1-2*lamb_)*w_
                b += yi
            else:
                w_ = w_
                W_ = w_*(1-2*lamb_)
                b = b
    return W_,b


Model1r4 = PerceptronTrainReg4(X_trainer, yv1_train)
Model2r4 = PerceptronTrainReg4(X_trainer, yv2_train)
Model3r4 = PerceptronTrainReg4(X_trainer, yv3_train)

Mode1r4 = Model1r4[0][:4]
Mode2r4 = Model2r4[0][:4]
Mode3r4 = Model3r4[0][:4]

#get Y labels
y_label1r4= PerceptronTest(-4.0, Mode1r4, X_trainer)
y_label2r4= PerceptronTest(-4.0, Mode2r4, X_trainer)
y_label3r4= PerceptronTest(-1.0, Mode3r4, X_trainer)

ylabel1r4 = y_label1r4
weight1r4 = ylabel1r4[0]
lab1r4 = ylabel1r4[1]
ylabel2r4 = y_label2r4
weight2r4 = ylabel2r4[0]
lab2r4 = ylabel2r4[1]
ylabel3r4 = y_label3r4
weight3r4 = ylabel3r4[0]
lab3r4 = ylabel3r4[1]

#multiclassifier at 0.01 lambda L2 Regularisation
Xconcr4 = np.vstack(([[weight1r4], [weight2r4], [weight3r4]]))
multiclassifierr4 = np.argmax(Xconcr4, axis=0)
multiclassr4 = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifierr4]

Accuracym4 = accuracy(multiclassr4, trainclassr)


print(f"The Weighted Vector and Bias for Class 1 Train Multiclass Dataset with L2 Regularization at lambda 10.0 is: {Model1r4}")
print(f"The Weighted Vector and Bias for Class 2 Train Multiclass Dataset with L2 Regularization at lambda 10.0 is: {Model2r4}")
print(f"The Weighted Vector and Bias for Class 3 Train Multiclass Dataset with L2 Regularization at lambda 10.0 is: {Model3r4}")
print(f"The accuracy score for Multiclass Train with L2 Regularization at lambda 10.0 is: {Accuracym4}")

#Regularization with lambda 100.0
def PerceptronTrainReg5(X_train, y_train, MaxIter=20):
    w_ = np.zeros(X_train.shape[1]) 
    b=0
    lamb_ = 100.0

    
    for _ in range(MaxIter):
        for xi,yi in zip(X_train, y_train):
            a = (np.dot(xi, w_)+b) 
            if yi*a <= 0:
                w_ +=yi*xi
                W_ = (1-2*lamb_)*w_
                b += yi
            else:
                w_ = w_
                W_ = w_*(1-2*lamb_)
                b = b
    return W_,b


Model1r5 = PerceptronTrainReg5(X_trainer, yv1_train)
Model2r5 = PerceptronTrainReg5(X_trainer, yv2_train)
Model3r5 = PerceptronTrainReg5(X_trainer, yv3_train)

Mode1r5 = Model1r5[0][:4]
Mode2r5 = Model2r5[0][:4]
Mode3r5 = Model3r5[0][:4]

#get Y labels
y_label1r5= PerceptronTest(-4.0, Mode1r5, X_trainer)
y_label2r5= PerceptronTest(-4.0, Mode2r5, X_trainer)
y_label3r5= PerceptronTest(-1.0, Mode3r5, X_trainer)

ylabel1r5 = y_label1r5
weight1r5 = ylabel1r5[0]
lab1r5 = ylabel1r5[1]
ylabel2r5 = y_label2r5
weight2r5 = ylabel2r5[0]
lab2r5 = ylabel2r5[1]
ylabel3r5 = y_label3r5
weight3r5 = ylabel3r5[0]
lab3r5 = ylabel3r5[1]

#multiclassifier at 0.01 lambda L2 Regularisation
Xconcr5 = np.vstack(([[weight1r5], [weight2r5], [weight3r5]]))
multiclassifierr5 = np.argmax(Xconcr5, axis=0)
multiclassr5 = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifierr5]

Accuracym5 = accuracy(multiclassr5, trainclassr)


print(f"The Weighted Vector and Bias for Class 1 Train Multiclass Dataset with L2 Regularization at lambda 100.0 is: {Model1r5}")
print(f"The Weighted Vector and Bias for Class 2 Train Multiclass Dataset with L2 Regularization at lambda 100.0 is: {Model2r5}")
print(f"The Weighted Vector and Bias for Class 3 Train Multiclass Dataset with L2 Regularization at lambda 100.0 is: {Model3r5}")
print(f"The accuracy score for Multiclass Train with L2 Regularization at lambda 100.0 is: {Accuracym5}")

test = pd.read_csv('test.data', header=None)

#Test dataset
#Regularization with lambda 0.01
Modeltrt = PerceptronTrainReg(X_tester, yv1_test)
Modelt2rt = PerceptronTrainReg(X_tester, yv2_test)
Modelt3rt = PerceptronTrainReg(X_tester, yv3_test)


Modeltrt1 = Modeltrt[0][:4]
Modelt2rt1 = Modelt2rt[0][:4]
Modelt3rt1 = Modelt3rt[0][:4]

#get Y labels
y_label1trt= PerceptronTest(0.0, Modeltrt1, X_tester)
y_label2trt= PerceptronTest(-4.0, Modelt2rt1, X_tester)
y_label3trt= PerceptronTest(-1.0, Modelt3rt1, X_tester)

ylabel1trt = y_label1trt
weight1trt = ylabel1trt[0]
lab1trt = ylabel1trt[1]
ylabel2trt = y_label2trt
weight2trt = ylabel2trt[0]
lab2trt = ylabel2trt[1]
ylabel3trt = y_label3trt
weight3trt = ylabel3trt[0]
lab3trt = ylabel3trt[1]

#multiclassifier at 0.01 lambda L2 Regularisation
Xconctrt = np.vstack(([[weight1trt], [weight2trt], [weight3trt]]))
multiclassifiertrt = np.argmax(Xconctrt, axis=0)
multiclasstrt = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifiertrt]

dataframe1trt = test.rename(columns={4:'Classes'}, inplace=True)
y_testmrt = test.loc[0:59,'Classes']
Accuracymtrrt = accuracy(multiclasstrt, y_testmrt)


print(f"The Weighted Vector and Bias for Class 1 Test Multiclass Dataset with L2 Regularization at lambda 0.01 is: {Modeltrt}")
print(f"The Weighted Vector and Bias for Class 2 Test Multiclass Dataset with L2 Regularization at lambda 0.01 is: {Modelt2rt}")
print(f"The Weighted Vector and Bias for Class 3 Test Multiclass Dataset with L2 Regularization at lambda 0.01 is: {Modelt3rt}")
print(f"The accuracy score for Multiclass Test with L2 Regularization at lambda 0.01 is: {Accuracymtrrt}")

#Regularization with lambda 0.1
Modeltrt2 = PerceptronTrainReg2(X_tester, yv1_test)
Modelt2rt2 = PerceptronTrainReg2(X_tester, yv2_test)
Modelt3rt2 = PerceptronTrainReg2(X_tester, yv3_test)

Modeltrt22 = Modeltrt2[0][:4]
Modelt2rt22 = Modelt2rt2[0][:4]
Modelt3rt22 = Modelt3rt2[0][:4]

#get Y labels
y_label1trt2= PerceptronTest(0.0, Modeltrt22, X_tester)
y_label2trt2= PerceptronTest(-4.0, Modelt2rt22, X_tester)
y_label3trt2= PerceptronTest(-1.0, Modelt3rt22 , X_tester)

ylabel1trt2 = y_label1trt2
weight1trt2 = ylabel1trt2[0]
lab1trt2 = ylabel1trt2[1]
ylabel2trt2 = y_label2trt2
weight2trt2 = ylabel2trt2[0]
lab2trt2 = ylabel2trt2[1]
ylabel3trt2 = y_label3trt2
weight3trt2 = ylabel3trt2[0]
lab3trt2 = ylabel3trt2[1]

#multiclassifier at 0.1 lambda L2 Regularisation
Xconctrt2 = np.vstack(([[weight1trt2], [weight2trt2], [weight3trt2]]))
multiclassifiertrt2 = np.argmax(Xconctrt2, axis=0)
multiclasstrt2 = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifiertrt2]

Accuracymtrrt2 = accuracy(multiclasstrt2, y_testmrt)


print(f"The Weighted Vector and Bias for Class 1 Test Multiclass Dataset with L2 Regularization at lambda 0.1 is: {Modeltrt2}")
print(f"The Weighted Vector and Bias for Class 2 Test Multiclass Dataset with L2 Regularization at lambda 0.1 is: {Modelt2rt2}")
print(f"The Weighted Vector and Bias for Class 3 Test Multiclass Dataset with L2 Regularization at lambda 0.1 is: {Modelt3rt2}")
print(f"The accuracy score for Multiclass Test with L2 Regularization at lambda 0.1 is: {Accuracymtrrt2}")

#Regularization with lambda 1.0
Modeltrt3 = PerceptronTrainReg3(X_tester, yv1_test)
Modelt2rt3 = PerceptronTrainReg3(X_tester, yv2_test)
Modelt3rt3 = PerceptronTrainReg3(X_tester, yv3_test)

Modeltrt33 = Modeltrt3[0][:4]
Modelt2rt33 = Modelt2rt3[0][:4]
Modelt3rt33 = Modelt3rt3[0][:4]

#get Y labels
y_label1trt3= PerceptronTest(0.0, Modeltrt33, X_tester)
y_label2trt3= PerceptronTest(-2.0, Modelt2rt33, X_tester)
y_label3trt3= PerceptronTest(-2.0, Modelt3rt33, X_tester)

ylabel1trt3 = y_label1trt3
weight1trt3 = ylabel1trt3[0]
lab1trt3 = ylabel1trt3[1]
ylabel2trt3 = y_label2trt3
weight2trt3 = ylabel2trt3[0]
lab2trt3 = ylabel2trt3[1]
ylabel3trt3 = y_label3trt3
weight3trt3 = ylabel3trt3[0]
lab3trt3 = ylabel3trt3[1]

#multiclassifier at 0.1 lambda L2 Regularisation
Xconctrt3 = np.vstack(([[weight1trt3], [weight2trt3], [weight3trt3]]))
multiclassifiertrt3 = np.argmax(Xconctrt3, axis=0)
multiclassifiertrt3
multiclasstrt3 = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifiertrt3]
Accuracymtrrt3 = accuracy(multiclasstrt3, y_testmrt)



print(f"The Weighted Vector and Bias for Class 1 Test Multiclass Dataset with L2 Regularization at lambda 1.0 is: {Modeltrt3}")
print(f"The Weighted Vector and Bias for Class 2 Test Multiclass Dataset with L2 Regularization at lambda 1.0 is: {Modelt2rt3}")
print(f"The Weighted Vector and Bias for Class 3 Test Multiclass Dataset with L2 Regularization at lambda 1.0 is: {Modelt3rt3}")
print(f"The accuracy score for Multiclass Test with L2 Regularization at lambda 1.0 is: {Accuracymtrrt3}")

#Regularization with lambda 10.0
Modeltrt4 = PerceptronTrainReg4(X_tester, yv1_test)
Modelt2rt4 = PerceptronTrainReg4(X_tester, yv2_test)
Modelt3rt4 = PerceptronTrainReg4(X_tester, yv3_test)

Modeltrt44 = Modeltrt4[0][:4]
Modelt2rt44 = Modelt2rt4[0][:4]
Modelt3rt44 = Modelt3rt4[0][:4]

#get Y labels
y_label1trt4= PerceptronTest(0.0, Modeltrt44, X_tester)
y_label2trt4= PerceptronTest(-4.0, Modelt2rt44, X_tester)
y_label3trt4= PerceptronTest(-1.0, Modelt3rt44, X_tester)

ylabel1trt4 = y_label1trt4
weight1trt4 = ylabel1trt4[0]
lab1trt4 = ylabel1trt4[1]
ylabel2trt4 = y_label2trt4
weight2trt4 = ylabel2trt4[0]
lab2trt4 = ylabel2trt4[1]
ylabel3trt4 = y_label3trt4
weight3trt4 = ylabel3trt4[0]
lab3trt4 = ylabel3trt4[1]

#multiclassifier at 0.1 lambda L2 Regularisation
Xconctrt4 = np.vstack(([[weight1trt4], [weight2trt4], [weight3trt4]]))
multiclassifiertrt4 = np.argmax(Xconctrt4, axis=0)
multiclasstrt4 = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifiertrt4]
Accuracymtrrt4 = accuracy(multiclasstrt4, y_testmrt)



print(f"The Weighted Vector and Bias for Class 1 Test Multiclass Dataset with L2 Regularization at lambda 10.0 is: {Modeltrt4}")
print(f"The Weighted Vector and Bias for Class 2 Test Multiclass Dataset with L2 Regularization at lambda 10.0 is: {Modelt2rt4}")
print(f"The Weighted Vector and Bias for Class 3 Test Multiclass Dataset with L2 Regularization at lambda 10.0 is: {Modelt3rt4}")
print(f"The accuracy score for Multiclass Test with L2 Regularization at lambda 10.0 is: {Accuracymtrrt4}")


#Regularization with lambda 100.0
Modeltrt5 = PerceptronTrainReg5(X_tester, yv1_test)
Modelt2rt5 = PerceptronTrainReg5(X_tester, yv2_test)
Modelt3rt5 = PerceptronTrainReg5(X_tester, yv3_test)

Modeltrt55 = Modeltrt5[0][:4]
Modelt2rt55 = Modelt2rt5[0][:4]
Modelt3rt55 = Modelt3rt5[0][:4]

#get Y labels
y_label1trt5= PerceptronTest(0.0, Modeltrt55, X_tester)
y_label2trt5= PerceptronTest(-4.0, Modelt2rt55, X_tester)
y_label3trt5= PerceptronTest(-1.0, Modelt3rt55, X_tester)

ylabel1trt5 = y_label1trt5
weight1trt5 = ylabel1trt5[0]
lab1trt5 = ylabel1trt5[1]
ylabel2trt5 = y_label2trt5
weight2trt5 = ylabel2trt5[0]
lab2trt5 = ylabel2trt5[1]
ylabel3trt5 = y_label3trt5
weight3trt5 = ylabel3trt5[0]
lab3trt5 = ylabel3trt5[1]

#multiclassifier at 0.1 lambda L2 Regularisation
Xconctrt5 = np.vstack(([[weight1trt5], [weight2trt5], [weight3trt5]]))
multiclassifiertrt5 = np.argmax(Xconctrt5, axis=0)
multiclasstrt5 = ['class-1' if i == 0 else 'class-2' if i ==1 else 'class-3' for i in multiclassifiertrt5]
Accuracymtrrt5 = accuracy(multiclasstrt5, y_testmrt)


print(f"The Weighted Vector and Bias for Class 1 Test Multiclass Dataset with L2 Regularization at lambda 100.0 is: {Modeltrt5}")
print(f"The Weighted Vector and Bias for Class 2 Test Multiclass Dataset with L2 Regularization at lambda 100.0 is: {Modelt2rt5}")
print(f"The Weighted Vector and Bias for Class 3 Test Multiclass Dataset with L2 Regularization at lambda 100.0 is: {Modelt3rt5}")
print(f"The accuracy score for Multiclass Test with L2 Regularization at lambda 100.0 is: {Accuracymtrrt5}")


# In[ ]:




