from __future__ import print_function
import numpy as np
import cv2
import pickle
import os
import sklearn.preprocessing
import sys
from random import randint


indexpath=0

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)

  return tracker

#geeting haar features
def haar_data(face):
   data=[]
   data2=[]
   data1=[]
   img = cv2.resize(face,(64,64))
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
   return data2


#the data labels
Hogsvmlable=cv2.ml.SVM_load("hogsvmtest.dat")
hoggendersvm=cv2.ml.SVM_load('hoggendersvmtest.dat')
haarsvm=cv2.ml.SVM_load('haarsvmtest.dat')
hoghaarsvm=cv2.ml.SVM_load('hog&haarsvmtrain.dat')
with open("hogknn.pkl", 'rb') as file:
    Hog_knn = pickle.load(file)


#data for names
names=['Anwar','Sabry','Gasser','Ayman','Essam','Nermen','sara']
gender=['Female','male']

hog = cv2.HOGDescriptor()


# Set video to load
face_cascade=cv2.CascadeClassifier('C:/Users/Gasser/Anaconda3/envs/vision/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
videoPath = "F:/faculty/4th/semester 1/computer vision/project/video/4.mp4"

# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)

#getting faces data and numbers
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7,minSize=(1,1),maxSize=(60,60))


## Select lists to use
bboxes = []
colors = []
nameslist=[]
genderlist=[]
path1=[]
pathcolor=[]

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
for (x, y, w, h) in faces:
  # draw bounding boxes over objects
  # selectROI's default behaviour is to draw box starting from the center
  # when fromCenter is set to false, you can draw box starting from top left corner
  face = np.copy(frame[y:y+h, x:x+w])
  face= cv2.resize(face, (64, 128))
        #""""""""""getting data""""""'
  hog_data=hog.compute(face)
  haardata=haar_data(face)

  a = np.array(haardata, np.float32)
  b = hog_data.reshape(hog_data.shape[0])
  b = np.array(b, np.float32)
  data = np.concatenate((b, a), axis=None)
  data = data.reshape(1, data.shape[0])
  #data  for drawing lines
  x1=(int)(x+x+w)/2
  y1=(int)(y+y+h)/2
  path1.append(x1)
  path1.append(y1)

  hog_data=hog_data.reshape(1,hog_data.shape[0])
  haardata=haardata.reshape(1,haardata.shape[0])

  res=Hogsvmlable.predict(hog_data)
  resg=hoggendersvm.predict(hog_data)
  ii=int(res[1][0])
  iii=int(resg[1][0])
  lab=names[ii]
  gen=gender[iii]
  nameslist.append(lab)
  genderlist.append(gen)
  bboxes.append((x,y,w,h))
  x1=randint(0, 255)
  x2=randint(0, 255)
  x3= randint(0, 255)
  pathcolor.append(x1)
  pathcolor.append(x2)
  pathcolor.append(x3)
  colors.append((x1,x2,x3))



#print('Selected bounding boxes {}'.format(bboxes))

# Specify the tracker type
trackerType = "CSRT"

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
  multiTracker.add(createTrackerByName(trackerType), frame, bbox)

# Process video and track objects
f=0
while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  f=f+1
  print(f)
  if(f%50==0):
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7,minSize=(1,1),maxSize=(60,60))


      ## Select boxes
      bboxes = []
      colors = []
      nameslist=[]
      genderlist=[]
      path1=[]
      pathcolor=[]

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
      for (x, y, w, h) in faces:
  # draw bounding boxes over objects
  # selectROI's default behaviour is to draw box starting from the center
  # when fromCenter is set to false, you can draw box starting from top left corner
       face = np.copy(frame[y:y+h, x:x+w])
       face= cv2.resize(face, (64, 128))
        #""""""""""getting data""""""'
       hog_data=hog.compute(face)
        # genderdata=hog.compute(face)
       haardata=haar_data(face)

       a = np.array(haardata, np.float32)
       b = hog_data.reshape(hog_data.shape[0])
       b = np.array(b, np.float32)
       data = np.concatenate((b, a), axis=None)
       data = data.reshape(1, data.shape[0])
       x1=(int)(x+x+w)/2
       y1=(int)(y+y+h)/2
       path1.append(x1)
       path1.append(y1)

      hog_data=hog_data.reshape(1,hog_data.shape[0])
      haardata=haardata.reshape(1,haardata.shape[0])

      res=Hogsvmlable.predict(hog_data)
      resg=hoggendersvm.predict(hog_data)
      ii=int(res[1][0])
      iii=int(resg[1][0])
      lab=names[ii]
      gen=gender[iii]
      nameslist.append(lab)
      genderlist.append(gen)
      bboxes.append((x,y,w,h))
      x1=randint(0, 255)
      x2=randint(0, 255)
      x3= randint(0, 255)
      pathcolor.append(x1)
      pathcolor.append(x2)
      pathcolor.append(x3)
      colors.append((x1,x2,x3))



#print('Selected bounding boxes {}'.format(bboxes))

# Specify the tracker type
      trackerType = "CSRT"

# Create MultiTracker object
      multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
      for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)







  # get updated location of objects in subsequent frames


  if f%1==0:

   success, boxes = multiTracker.update(frame)

  # draw tracked objects
   for i, newbox in enumerate(boxes):
     p1 = (int(newbox[0]), int(newbox[1]))
     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
     cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
     cv2.putText(frame,nameslist[i],(p1[0],p1[1]-5),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
     cv2.putText(frame,genderlist[i],(p1[0],p1[1]+50+5),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
     x1=(int)(newbox[0]+newbox[0]+newbox[2])/2
     y1=(int)(newbox[1]+newbox[1]+newbox[3])/2
     path1.append(x1)
     path1.append(y1)


   gass=len(path1)
   print (len(path1))
   rr=len(boxes)*4
   indexpath=0
   co=0
   gass=gass-(len(boxes)*2)-1
   path1=[round(x) for x in path1]
   bas=0


   # draw lines on screen
   #if(f==49):
   while(indexpath < len(path1) ):
       u=path1[indexpath]
       y=path1[indexpath+1]
       x1=pathcolor[bas]
       x2=pathcolor[bas+1]
       x3=pathcolor[bas+2]
       cv2.circle(frame,(u,y),2,(x1,x2,x3),2)
       bas=bas+3
       if(bas==len(pathcolor)):
           bas=0
       indexpath=indexpath+2
       co=co+1
           # yo=0
           # while(yo<f-1):
           #           u=path1[indexpath]
           #           y=path1[indexpath+1]
           #           print(indexpath+(len(boxes)*2))
           #           print(len(path1))
           #           u1=path1[indexpath+(len(boxes)*2)]
           #           y1=path1[indexpath+(len(boxes)*2)+1]
           #           # frame[u,y]=[0,0,0]
           #           x1=pathcolor[bas]
           #           x2=pathcolor[bas+1]
           #           x3=pathcolor[bas+2]
           #           cv2.line(frame,(u,y),(u1,y1),(x1,x2,x3),5)
           #           indexpath=indexpath+2
           #           bas=bas+3
           #           yo=yo+1
           # co=co+1


#show frame
   cv2.imshow('MultiTracker', frame)
   cv2.imwrite("frame%d.jpg" % f, frame)


   #quit on ESC button
   if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
     break
