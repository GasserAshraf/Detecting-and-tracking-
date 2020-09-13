import glob
import os
import pickle
import numpy as np
import cv2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

hog = cv2.HOGDescriptor()
data1=[]
labels1=[]

for f in glob.glob('E:/female/*.jpg'):
 print('new female image')
 img=cv2.imread(f)
 img=cv2.resize(img,(64,128))
 feat=hog.compute(img)
 data1.append(feat)
 labels1.append(0)


for f in glob.glob('E:/male/*.jpg'):
 img=cv2.imread(f)
 print('new male image')
 img=cv2.resize(img,(64,128))
 feat=hog.compute(img)
 data1.append(feat)
 labels1.append(1)


print(len(data1))
print (len(labels1))
d=np.array(data1,np.float32)
l=np.array(labels1)
d=d.reshape(d.shape[0],d.shape[1]*d.shape[2])
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

print('start training')

svm.train(d, cv2.ml.ROW_SAMPLE, l)
print('finish')
svm.save('hoggendersvmtest.dat')
print(len(data1))
print('Done')
