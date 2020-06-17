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
import dlib
import threading
import math

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', 
                help = 'path to yolo config file', default=r'yolov3_custom.cfg')
ap.add_argument('-w', '--weights', 
                help = 'path to yolo pre-trained weights', default=r'yolov3_custom_last.weights')
ap.add_argument('-cl', '--classes', 
                help = 'path to text file containing class names',default=r'obj.names')
args = ap.parse_args()


# Load names classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

#Generate color for each class randomly
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Define network from configuration file and load the weights from the given weights file
net = cv2.dnn.readNet(args.weights,args.config)


nv = []         #total number of persons viewed the advertisement
count=0
temp=[]
total_p=[]      #total number of persons passed 
c3 = 0          #final male count
c4 = 0          #final female count

OUTPUT_SIZE_WIDTH = 720
OUTPUT_SIZE_HEIGHT = 720                                                                                            #--------------------------------------------

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Darw a rectangle surrounding the object and its class name 
def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    #print("label",label)
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def FindPoint(left, top,right, bottom, cx, cy) :
    if (cx > left and cx < right and 
        cy > top and cy < bottom) :
        return True
    else : 
        return False

    
def  centre(top,right,bottom,left):
    y = bottom + int((top-bottom)*0.5)
    x = left + int((right - left)*0.5)
    return  [x,y]

class VideoTracker(object):
    def __init__(self, cfg):
        use_cuda = torch.cuda.is_available()
        
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
            global c3 
            global c4
            ret,frame = cap.read()
            frame=cv2.flip(frame,1)
            Width = frame.shape[1]
            Height = frame.shape[0]
            blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))

            c1 = 0          #real-time female count
            c2 = 0          #real-time male count
            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4

            for out in outs: 
        #print(out.shape)
                for detection in out:
            
        #each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
                    scores = detection[5:]#classes scores starts from index 5
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        #print("confidence",confidence)
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        #print("class_ids",class_ids)
                        confidences.append(float(confidence))
                        #print("confidences",confidences)
                        boxes.append([x, y, w, h])
    
    # apply  non-maximum suppression algorithm on the bounding boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
            for i in indices:
                i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                if class_ids[i] == 1:
                    c1 = c1 + 1
                elif class_ids[i] == 0:
                    c2 = c2 + 1
                else:
                    pass
                print("Num of live female viewers:",c1)
                print("Num of live male viewers:",c2)
                draw_pred(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

            face_locations = face_recognition.face_locations(frame)

            frameCounter += 1       #------------------------------------------------------------------------------
            start = time.time()
 
            bbox_xywh, cls_conf, cls_ids = self.detector(frame)
            #cv2.putText(frame,str(bbox_xywh),(0,15), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255) ,2)
            if bbox_xywh is not None:
                count=0
                
                mask = cls_ids==0
                bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small

                cls_conf = cls_conf[mask]

                outputs = self.deepsort.update(bbox_xywh, cls_conf, frame)  #left,top,right,bottom
                face_locations = face_recognition.face_locations(frame)
                #print("fl",face_locations)
                #for top, right, bottom, left in face_locations:
                 #   cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) #for face detector box
                
                     
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
                    frame = draw_boxes(frame, bbox_xyxy, identities)   #for tracker box

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
                                    if class_ids[i] == 1:
                                        c4 = c4 + 1
                                    elif class_ids[i] == 0:
                                        c3 = c3 + 1
                                    else:
                                        pass

                                    
            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))
                                   
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #print(count)  
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

