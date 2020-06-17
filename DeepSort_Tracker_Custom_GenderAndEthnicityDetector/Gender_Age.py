import os
import cv2
import time
import argparse
import torch
import numpy as np
import face_recognition
from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
import dlib                                                                                                                             #--------------------------------------------
import threading
import math

nv = []
count=0
temp=[]
total_p=[]
gender = []

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #-------------------------------------------

OUTPUT_SIZE_WIDTH = 720
OUTPUT_SIZE_HEIGHT = 720                                                                                            #--------------------------------------------

parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto=r"opencv_face_detector.pbtxt"
faceModel=r"opencv_face_detector_uint8.pb"
ageProto=r"age_deploy.prototxt"
ageModel=r"age_net.caffemodel"
genderProto=r"gender_deploy.prototxt"
genderModel=r"gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-4)', '(6-10)', '(11-15)', '(16-20)', '(21-25)', '(26-30)', '(31-35)', '(36-40)', '(41-45)', '(46-50)', '(51-55)','(56-60)','(61-65)','(66-70)','(71-75)','(76-80)','(81-85)','(86-90)','(91-95)','(96-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

c3=0        #final male count
c4=0        #final female count
info = []
padding=20

def FindPoint(left, top,right, bottom, cx, cy) :
    if (cx > left and cx < right and 
        cy > top and cy < bottom) :
        return True
    else : 
        return False
    
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])

    return frameOpencvDnn,faceBoxes

def func(x,y,faceBoxes,frame):
    try:
        global c1
        global c2
        global gender
        global fc
        c1=0
        c2=0
        for fc in faceBoxes:
            face=frame[max(0,fc[1]-padding):
                       min(fc[3]+padding,frame.shape[0]-1),max(0,fc[0]-padding)
                       :min(fc[2]+padding, frame.shape[1]-1)]
            
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            
        return gender
    except Exception as e:
        print("Exception while finding gender : ", e)
        return None

def  centre(top,right,bottom,left):
    y = bottom + int((top-bottom)*0.5)
    x = left + int((right - left)*0.5)
    return  [x,y]

class VideoTracker(object):
    def __init__(self, cfg):
        use_cuda = torch.cuda.is_available()
       # if not use_cuda:
        #    raise UserWarning("Running in cpu mode!")


        
        self.detector = build_detector(cfg, use_cuda=False)
        self.deepsort = build_tracker(cfg, use_cuda=False)
        self.class_names = self.detector.class_names



    
    def __enter__(self):
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def doRecognizePerson(faceNames, fid):    #------------------------------------------------------------------------------
        time.sleep(2)
        faceNames[ fid ] = "Person " + str(fid)         #-----------------------------------------------------------------------------

    #Start the window thread for the two windows we are using
    cv2.startWindowThread()

    #The color of the rectangle we draw around the face
    rectangleColor = (0,165,255)

    #variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0

    #Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}
      
    def track(self):
        count=0
        
        global temp
        
        cap = cv2.VideoCapture(0)

         #Start the window thread for the two windows we are using
        cv2.startWindowThread()

        #The color of the rectangle we draw around the face
        rectangleColor = (0,165,255)

        #variables holding the current frame number and the current faceid
        frameCounter = 0
        currentFaceID = 0

        #Variables holding the correlation trackers and the name per faceid
        faceTrackers = {}
        faceNames = {}

        
        while True:
            tk = []
            global c3
            global c4
            
            ret,frame = cap.read()
            frame=cv2.flip(frame,1)
            frame=cv2.resize(frame,(720,720))
            face_locations = face_recognition.face_locations(frame)
            

            frameCounter += 1       #------------------------------------------------------------------------------
            start = time.time()
            resultImg,faceBoxes=highlightFace(faceNet,frame)
            if not faceBoxes:
                print("No face detected")
 
            bbox_xywh, cls_conf, cls_ids = self.detector(frame)

            if bbox_xywh is not None:
                count=0
                c1 = 0              #live male count
                c2 = 0              #live female count
                
                    #print("Centre of face",temp)
                    
                mask = cls_ids==0
                bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small

                cls_conf = cls_conf[mask]

                outputs = self.deepsort.update(bbox_xywh, cls_conf, frame)  #left,top,right,bottom
                face_locations = face_recognition.face_locations(frame)
                #print("fl",face_locations)
                for top, right, bottom, left in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                for fc in faceBoxes:
                            face=frame[max(0,fc[1]-padding):
                                    min(fc[3]+padding,frame.shape[0]-1),max(0,fc[0]-padding)
                                    :min(fc[2]+padding, frame.shape[1]-1)]
                            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                            genderNet.setInput(blob)
                            genderPreds=genderNet.forward()
                            gender=genderList[genderPreds[0].argmax()]
                            ageNet.setInput(blob)
                            agePreds=ageNet.forward()
                            age=ageList[agePreds[0].argmax()]

                                
                            if gender=='Male':
                                c1 = c1+1
                            else:
                                c2 = c2 +1
                            print('Male detected ',c1)
                            print('Female detected',c2)
                                
                     
                print("No of live viewers: ",len(face_locations))
                temp1=[]
                for i in range(len(face_locations)):
                    t1,t2,t3,t4=face_locations[i]
                    temp=centre(t1,t2,t3,t4)
                    temp1.append(temp)                
                #print("Outputs: ",outputs)
                #print(len(outputs))
                for i in  range(len(outputs)):
                 if outputs[i][4] not in total_p:
                     total_p.append(outputs[i][4])
                     #print(temp)
                print("Total No of people: ",len(outputs))
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:-1]
                    identities = outputs[:,-1]
                    frame = draw_boxes(frame, bbox_xyxy, identities)

                for i in range(len(outputs)):
                    if(outputs[i][4] not in nv):
                        for j in range(len(temp1)):
                            a,b,c,d,e= outputs[i]
                            flag=FindPoint(a,b,c,d,temp1[j][0],temp1[j][1])
                        #print(flag)
                            if flag:
                                if e not in nv:
                                    nv.append(e)
                                    time.sleep(4)
                                    gen = func(temp1[j][0],temp1[j][1],faceBoxes,frame)
                                #print("Answer",gender)
                                    if gen=='Male':
                                        c3 = c3+1
                                    elif gen == 'Female':
                                        c4 = c4 +1
                                    else:
                                        gen = None
            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))
            label ="Age:"+age           
            cv2.putText(frame, label, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, 200)
            cv2.imshow("frame", frame)


            #cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()            

        return [nv,total_p,c3,c4]

if __name__=="__main__":
    cfg = get_config()
    cfg.merge_from_file("./configs/yolov3.yaml")
    cfg.merge_from_file("./configs/deep_sort.yaml")

    with VideoTracker(cfg) as vdo_trk:
        p = vdo_trk.track()
        print("Total no of person who viewed advertisement "+str(len(nv))+" \n Total no of persons who passed by the advertisement board "+str(len(total_p)))
        print('Total number of Males viewed the advertisement',c3)
        print('Total number of Females viewed the advertisement',c4)

