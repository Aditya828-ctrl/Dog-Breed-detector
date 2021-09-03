#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image                  
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
import cv2 
import matplotlib.pyplot as plt                        
from glob import glob   
from keras.models import load_model 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def makeTensor(file):
    img=image.load_img(file,target_size=(224,224))
    return np.expand_dims(image.img_to_array(img),axis=0)


# In[3]:


def ResNet50Predictions(file):
    img = preprocess_input(makeTensor(file))
    m=ResNet50(weights='imagenet')
    x=m(img)
    x=np.argmax(x)
    return x
                

# In[4]:


def isDogDetected(file):
    prediction = ResNet50Predictions(file)
    return ((prediction <= 268) & (prediction >= 151))


# In[5]:


def setDetector():
    detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    return detector


# In[6]:


def isHumanFaceDetected(file):
    detector=setDetector()
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray)
    return len(faces) > 0


# In[7]:


def loadBreedDetectorModel():
    model = load_model('BreedDetector.h5')
    return model


# In[8]:


def loadBreedList():
    n=[]
    for i in sorted(glob("dogImages/train/*/")):
        n.append(i[20:-1])
    breedList=np.array(n)
    return breedList


# In[9]:


def getDogBreed(file):
    from keras.applications.vgg16 import VGG16, preprocess_input
    img=image.load_img(file,target_size=(224,224))
    x= np.expand_dims(image.img_to_array(img),axis=0)
    m=VGG16(weights='imagenet', include_top=False)
    x=m(preprocess_input(x))
    model=loadBreedDetectorModel()
    prediction=model(x)
    breedList=loadBreedList()
    breedName=breedList[np.argmax(prediction)]
    return breedName


# In[10]:


def displayInputImage(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


# In[11]:


def answer(file):
    dogBreed=getDogBreed(file)
    if(isDogDetected(file)):
        print("It is an image of dog.")
        print("Its breed: ", dogBreed)
    else:
        print("Its not a dog image.\nChecking for human faces in the image...")
        if(isHumanFaceDetected(file)):
            print("Human face(s) detected.\nThe person does look similar to a ", dogBreed)
        else:
            print("Sorry neither a dog nor a human.")
            
    print("Displaying the input image..")
    displayInputImage(file)

