# DeepSort_Tracker_Custom_GenderAndEthnicityDetector
This is a project previously named as AI/ML Based Intelligent Video Advertisement Analysis. Majorly used for getting insights when people 
are moving on road and analysing if they are viewing your hoarding on road or not. Simultaneously detecting their gender and ethnicity. For gender I trained 10K manually labelled by me images on YOLOV3 Darknet framework and then again labelled those for ethnicity wise and trained them again on Darknet framework to detect ethnicity. If you don't have your dataset then you can use caffe  model weights for gender and age detection. I have made a file for gender and age detection using caffe model weights. A file for separately detecting gender on my custom labelled dataset. A file for detecting ethnicity on my custom labelled dataset. I have also added a counter for real-time counting of people in frame, people actually viewing the advertisement, Number of male / female live viewing the advertisement and number of total male and female at the end of the day.