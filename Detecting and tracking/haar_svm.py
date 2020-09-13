import glob
import glob
import os
import pickle
import numpy as np
import cv2
import sklearn.preprocessing

data3=[]
labels=[]

for f in glob.glob("E:/neww0/*.jpg"):
 img = cv2.imread(f)
 img = cv2.resize(img,(64,64))
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 sklearn.preprocessing.normalize(gray, 'l2')
 data=[]
 data2=[]
 data1=[]
 for i in range(8):
  for j in range(8):
    block=gray[8*i:8*i+8,j*8:j*8+8]
    up=block[[0,1,2,3],:]
    down=block[[4,5,6,7],:]
    x=int(up.sum())
    y=int(down.sum())
    z=x-y
    left=block[:,[0,1,2]]
    midle=block[:,[3,4]]
    right=block[:,[5,6,7]]
    left1=int(left.sum())
    midle1=int(midle.sum())
    right1=int(right.sum())
    sum=left1-midle1+right1
    data.append(z)
    data1.append(sum)
 l=data
 l=np.array(l,np.float32)
 l1=data1
 l1=np.array(l1,np.float32)
 data2=np.concatenate((l, l1), axis=None)
 labels.append(0)

 data3.append(data2)

for f in glob.glob("E:/neww1/*.jpg"):
 data=[]
 data2=[]
 data1=[]
 img = cv2.imread(f)
 img = cv2.resize(img,(64,64))
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 sklearn.preprocessing.normalize(gray, 'l2')
 for i in range(8):
  for j in range(8):
    block=gray[8*i:8*i+8,j*8:j*8+8]
    up=block[[0,1,2,3],:]
    down=block[[4,5,6,7],:]
    x=int(up.sum())
    y=int(down.sum())
    z=x-y
    left=block[:,[0,1,2]]
    midle=block[:,[3,4]]
    right=block[:,[5,6,7]]
    left1=int(left.sum())
    midle1=int(midle.sum())
    right1=int(right.sum())
    sum=left1-midle1+right1
    data.append(z)
    data1.append(sum)
 l=data
 l=np.array(l,np.float32)
 l1=data1
 l1=np.array(l1,np.float32)
 data2=np.concatenate((l, l1), axis=None)
 labels.append(1)

 data3.append(data2)

for f in glob.glob("E:/neww2/*.jpg"):
 data=[]
 data2=[]
 data1=[]
 img = cv2.imread(f)
 img = cv2.resize(img,(64,64))
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 sklearn.preprocessing.normalize(gray, 'l2')
 for i in range(8):
  for j in range(8):
    block=gray[8*i:8*i+8,j*8:j*8+8]
    up=block[[0,1,2,3],:]
    down=block[[4,5,6,7],:]
    x=int(up.sum())
    y=int(down.sum())
    z=x-y
    left=block[:,[0,1,2]]
    midle=block[:,[3,4]]
    right=block[:,[5,6,7]]
    left1=int(left.sum())
    midle1=int(midle.sum())
    right1=int(right.sum())
    sum=left1-midle1+right1
    data.append(z)
    data1.append(sum)
 l=data
 l=np.array(l,np.float32)
 l1=data1
 l1=np.array(l1,np.float32)
 data2=np.concatenate((l, l1), axis=None)
 labels.append(2)

 data3.append(data2)

for f in glob.glob("E:/neww3/*.jpg"):
 data=[]
 data2=[]
 data1=[]
 img = cv2.imread(f)
 img = cv2.resize(img,(64,64))
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 sklearn.preprocessing.normalize(gray, 'l2')
 for i in range(8):
  for j in range(8):
    block=gray[8*i:8*i+8,j*8:j*8+8]
    up=block[[0,1,2,3],:]
    down=block[[4,5,6,7],:]
    x=int(up.sum())
    y=int(down.sum())
    z=x-y
    left=block[:,[0,1,2]]
    midle=block[:,[3,4]]
    right=block[:,[5,6,7]]
    left1=int(left.sum())
    midle1=int(midle.sum())
    right1=int(right.sum())
    sum=left1-midle1+right1
    data.append(z)
    data1.append(sum)
 l=data
 l=np.array(l,np.float32)
 l1=data1
 l1=np.array(l1,np.float32)
 data2=np.concatenate((l, l1), axis=None)
 labels.append(3)

 data3.append(data2)

for f in glob.glob("E:/neww4/*.jpg"):
 data=[]
 data2=[]
 data1=[]
 img = cv2.imread(f)
 img = cv2.resize(img,(64,64))
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 sklearn.preprocessing.normalize(gray, 'l2')
 for i in range(8):
  for j in range(8):
    block=gray[8*i:8*i+8,j*8:j*8+8]
    up=block[[0,1,2,3],:]
    down=block[[4,5,6,7],:]
    x=int(up.sum())
    y=int(down.sum())
    z=x-y
    left=block[:,[0,1,2]]
    midle=block[:,[3,4]]
    right=block[:,[5,6,7]]
    left1=int(left.sum())
    midle1=int(midle.sum())
    right1=int(right.sum())
    sum=left1-midle1+right1
    data.append(z)
    data1.append(sum)
 l=data
 l=np.array(l,np.float32)
 l1=data1
 l1=np.array(l1,np.float32)
 data2=np.concatenate((l, l1), axis=None)
 labels.append(4)

 data3.append(data2)


d=np.array(data3,np.float32)
l=np.array(labels)


svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

print(len(data3))
print(len(labels))
print('start training')
svm.train(d, cv2.ml.ROW_SAMPLE, l)
print('ending')
svm.save('haarsvmtest.dat')

print('Done1')

# print(len(data1))
# print(len(labels))
# print('Done')

