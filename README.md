# DeepSort_Tracker_Custom_GenderAndEthnicityDetector
This is a project previously named as AI/ML Based Intelligent Video Advertisement Analysis. Majorly used for getting insights when people are moving on road and analysing if they are viewing your hoarding on road or not. Simultaneously detecting their gender and ethnicity. For gender I trained 10K manually labelled by me images on YOLOV3 Darknet framework and then again labelled those for ethnicity wise and trained them again on Darknet framework to detect ethnicity. If you don't have your dataset then you can use caffe  model weights for gender and age detection. I have made a file for gender and age detection using caffe model weights. A file for separately detecting gender on my custom labelled dataset. A file for detecting ethnicity on my custom labelled dataset. I have also added a counter for real-time counting of people in frame, people actually viewing the advertisement, Number of male / female live viewing the advertisement and number of total male and female at the end of the day.


This is application made in order to detect how many number of people are actually viewing the advertisement board and how many that are just passing by without viewing into it. Get insights from it like gender, age ethnicity.

Futher improvement direction  
- Train detector on specific dataset rather than the official one.
- Retrain REID model on pedestrain dataset for better performance.
- Replace YOLOv3 detector with advanced ones.
- Detect hair color of people passing by, am currently working on it using Mask-RCNN

Any contributions to this repository is welcome!


## Tracker Introduction
This tracker is an implement of MOT tracking algorithm deep sort. Deep sort is basicly the same with sort but added a CNN model to extract features in image of human part bounded by a detector. This CNN model is indeed a RE-ID model and the detector used in [PAPER](https://arxiv.org/abs/1703.07402) is FasterRCNN , and the original source code is [HERE](https://github.com/nwojke/deep_sort).  
However in original code, the CNN model is implemented with tensorflow, which I'm not familier with. SO I re-implemented the CNN feature extraction model with PyTorch, and changed the CNN model a little bit. Also, I use **YOLOv3** to generate bboxes instead of FasterRCNN.

## Dependencies
- python 3 (python2 not sure)
- numpy
- scipy
- opencv-python
- sklearn
- torch >= 0.4
- torchvision >= 0.1
- pillow
- vizer
- edict

## Quick Start
0. Check all dependencies installed
```bash
pip install -r requirements.txt
```
for user in china, you can specify pypi source to accelerate install like:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 
```

1. Clone this repository
```
git clone <url>
```

2. Download YOLOv3 parameters and ethnicity and gender weights
```
cd detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
cd ../../../
```
Ethnicty weights link - 
Gender weights link - 
Add these into the main folder 
DeepSort_Tracker_Custom_GenderAndEthnicityDetector/yolov3_custom_eth_last.weights
DeepSort_Tracker_Custom_GenderAndEthnicityDetector/yolov3_custom_last.weights



## Training the model
The original model used in paper is in original_model.py, and its parameter here [original_ckpt.t7](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).  

To train the model, first you need download [Market1501](http://www.liangzheng.org/Project/project_reid.html) dataset or [Mars](http://www.liangzheng.com.cn/Project/project_mars.html) dataset.  

Then you can try [train.py](deep_sort/deep/train.py) to train your own parameter and evaluate it using [test.py](deep_sort/deep/test.py) and [evaluate.py](deep_sort/deep/evalute.py).
![train.jpg](deep_sort/deep/train.jpg)

I created my dataset of 10k images for detector purpose. For raining purpose I used the already trained model weights of yolo, these were trained on coco dataset.

## Demo videos and images
[demo.avi](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
[demo2.avi](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)

![1.jpg](demo/1.jpg)
![2.jpg](demo/2.jpg)

Later merged a detector with this tracker, for detector purpose I used initially used caffe model files already in the depository they are there.

Then I trained my gender detector on dataset of 10k images. I hand labelled those images collected from various sources and trained on YOLOV3 Darknet framework, it detected successfully, then labelled those images according to ethnicity eight groups - asian_woman, asian_man, indian_man, indian_woman, black_man, black_woman, white_man, white_woman. That also worked well. I have also added a counter for the purpose of counting people in realtime. 

Three Counters used -
1) Count people that are just sitting infront of the camera and not viewing into the camera or screen 
2) Count those people that are actually viewing into the camera
3) Count people gender wise

The counter gives count of realtime people in frame and also total number of people that passed infront of the camera without viewing into it,  number of people passed infront of the camera with viewing into it, number of man and woman at the end of the day. 

tracker_ethnicty.py - detects gender and ethnicity along with tracker
tracker_gender.py - detects only gender along with tracker
Gender_Age.py - Detects gender and age using caffe model files along with tracker

## References
- paper: [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

- code: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)

- paper: [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

- code: [Joseph Redmon/yolov3](https://pjreddie.com/darknet/yolo/)
