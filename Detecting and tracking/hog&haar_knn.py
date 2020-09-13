import glob
import os
import pickle
import numpy as np
import cv2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsClassifier

hog = cv2.HOGDescriptor()

data=[]
alldata=[]
labels=[]


for f in glob.glob("E:/neww0/*.jpg"):
    data=[]
    data2=[]
    data1=[]
    img2=cv2.imread(f)
    img=cv2.resize(img2,(64,128))
    img1=cv2.resize(img2,(64,64))
    gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    feat=hog.compute(img)
    sklearn.preprocessing.normalize(gray, 'l2')
    labels.append(0)
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
    feat=feat.reshape(feat.shape[0])
    feat=np.array(feat,np.float32)
    data=np.concatenate((feat, data2), axis=None)
    alldata.append(data)

for f in glob.glob("E:/neww1/*.jpg"):
    data=[]
    data2=[]
    data1=[]
    img2=cv2.imread(f)
    img=cv2.resize(img2,(64,128))
    img1=cv2.resize(img2,(64,64))
    gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    feat=hog.compute(img)
    sklearn.preprocessing.normalize(gray, 'l2')
    labels.append(1)
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
    feat=feat.reshape(feat.shape[0])
    feat=np.array(feat,np.float32)
    data=np.concatenate((feat, data2), axis=None)
    alldata.append(data)

for f in glob.glob("E:/neww2/*.jpg"):
    data=[]
    data2=[]
    data1=[]
    img2=cv2.imread(f)
    img=cv2.resize(img2,(64,128))
    img1=cv2.resize(img2,(64,64))
    gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    feat=hog.compute(img)
    sklearn.preprocessing.normalize(gray, 'l2')
    labels.append(2)
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
    feat=feat.reshape(feat.shape[0])
    feat=np.array(feat,np.float32)
    data=np.concatenate((feat, data2), axis=None)
    alldata.append(data)

for f in glob.glob("E:/neww3/*.jpg"):
    data=[]
    data2=[]
    data1=[]
    img2=cv2.imread(f)
    img=cv2.resize(img2,(64,128))
    img1=cv2.resize(img2,(64,64))
    gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    feat=hog.compute(img)
    sklearn.preprocessing.normalize(gray, 'l2')
    labels.append(3)
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
    feat=feat.reshape(feat.shape[0])
    feat=np.array(feat,np.float32)
    data=np.concatenate((feat, data2), axis=None)
    alldata.append(data)

for f in glob.glob("E:/neww4/*.jpg"):
    data=[]
    data2=[]
    data1=[]
    img2=cv2.imread(f)
    img=cv2.resize(img2,(64,128))
    img1=cv2.resize(img2,(64,64))
    gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    feat=hog.compute(img)
    sklearn.preprocessing.normalize(gray, 'l2')
    labels.append(4)
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
    feat=feat.reshape(feat.shape[0])
    feat=np.array(feat,np.float32)
    data=np.concatenate((feat, data2), axis=None)
    alldata.append(data)



d=np.array(alldata,np.float32)
l=np.array(labels)
print('Done1')

model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
model.fit(d,l)

with open('haar&hogk.pkl','wb') as  file:
    pickle.dump(model,file)

print('Done')

