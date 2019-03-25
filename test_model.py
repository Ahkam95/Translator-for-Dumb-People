#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
#from pygame import mixer # Load the required library
import requests
import time


# In[15]:


from keras.preprocessing.image import array_to_img, img_to_array
from keras.models import model_from_json


# In[16]:


# load network model and network weights from files
def read_model(network_path):
    exit_ifnex(network_path)
    model = model_from_json(open(os.path.join(network_path, 'architecture.json')).read())
    model.load_weights(os.path.join(network_path, 'weights.h5'))
    return model


# In[17]:


def max_index_of(array):
    m = -1
    index = -1
    for i in range(len(array)):
        if array[i] > m:
            m = array[i]
            index = i
    return index


# In[18]:


def exit_ifnex(directory):
    if not os.path.exists(directory):
        print(directory, 'does not exist')
        exit()
def timestamp():
    return int(round(time.time() * 1000))


# In[19]:


# load neural network
model = read_model('../model')
headers={'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36"}
width, height, channel = 300, 280, 1


# In[20]:


background = None

# Start with a halfway point between 0 and 1 of accumulated weight
accumulated_weight = 0.5


# Manually set up our ROI for grabbing the hand.
# Feel free to change these. I just chose the top right corner for filming.
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600


# In[21]:


def calc_accum_avg(frame, accumulated_weight):
    '''
    Given a frame and a previous accumulated weight, computed the weighted average of the image passed in.
    '''
    
    # Grab the background
    global background
    
    # For first time, create the background from a copy of the frame.
    if background is None:
        background = frame.copy().astype("float")
        return None

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(frame, background, accumulated_weight)


# In[22]:


def segment(frame, threshold=25):
    global background
    
    # Calculates the Absolute Differentce between the backgroud and the passed in frame
    diff = cv2.absdiff(background.astype("uint8"), frame)

    # Apply a threshold to the image so we can grab the foreground
    # We only need the threshold, so we will throw away the first item in the tuple with an underscore _
    ret , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours form the image
    # Again, only grabbing what we need here and throwing away the rest
    image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list is 0, then we didn't grab any contours!
    if len(contours) == 0:
        return None
    else:
        # Given the way we are using the program, the largest external contour should be the hand (largest by area)
        # This will be our segment
        hand_segment = max(contours, key=cv2.contourArea)
        
        # Return both the hand segment and the thresholded hand image
        return (thresholded, hand_segment)


# In[2]:


from time import sleep
from threading import *
from pygame import mixer # Load the required library


# In[2]:


class play(Thread):
    def run(self):
        global text
        global i
        while True:
            if len(text)>10:
                params={'ie':'UTF-8','q': text,'tl': 'en','client': 'gtx'}
                r= requests.get(url,params=params,headers=headers)
                
                with open(f"thread{i}.mp3",'wb') as f:
                    f.write(r.content)
                mixer.init()
                mixer.music.load(f"thread{i}.mp3")
                mixer.music.play()
                if i==0:
                    if os.path.exists("thread1.mp3"):
                        os.remove("thread1.mp3")
                    i=1
                else:
                    if os.path.exists("thread0.mp3"):
                        os.remove("thread0.mp3")
                    i=0
                print(text)
                text=""
                


# In[ ]:





# In[24]:


cam = cv2.VideoCapture(0)
# capture settings
time_between_capture_ms = 1000
last_capture = timestamp()
text=""
# Intialize a frame count
num_frames = 0

while True:
    # get the current frame
    ret, frame = cam.read()

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    frame_copy = frame.copy()

    # Grab the ROI from the frame
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    # Apply grayscale and blur to ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # For the first 30 frames we will calculate the average of the background.
    # We will tell the user while this is happening
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Finger Count",frame_copy)
            
    else:
        # now that we have the background, we can segment the hand.
        
        # segment the hand region
        hand = segment(gray)

        # First check if we were able to actually detect a hand
        if hand is not None:
            
            # unpack
            thresholded, hand_segment = hand

            # Draw contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)

            # Count the fingers
            #fingers = count_fingers(thresholded, hand_segment)
            
            # Display count
            #cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Also display the thresholded image
            cv2.imshow("Thesholded", thresholded)
            image = cv2.resize(thresholded, (width, height))
            image = img_to_array(image)
            image = np.array(image, dtype="float") / 255.0
            image = image.reshape(1, width, height, channel)
            if timestamp()-last_capture>time_between_capture_ms:
                output = model.predict(image)
                os.system('clear')

                if max_index_of(output[0])==0:
                    text=text+" I" #paper
                if max_index_of(output[0])==1:
                    text=text+" am"
                if max_index_of(output[0])==2:
                    text=text+" human"
                print(max_index_of(output[0]))   
                last_capture = timestamp()
            
    # Draw ROI Rectangle on frame copy
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)

    # increment the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.imshow("Finger Count", frame_copy)


    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()


# In[ ]:




