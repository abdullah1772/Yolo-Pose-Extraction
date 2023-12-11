import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pose_extract import Extrac_Pose

model = YOLO('../Pose_Extraction/yolov8l-pose.pt')

video = cv2.VideoCapture('../Pose_Extraction/videos/front.mp4')

while video.isOpened():

    ret, frame = video.read()

    if ret:
        frame = cv2.resize(frame, (600,400))
        
        results = model(frame)
        anot_frame = results[0].plot()
    
        keypoints = results[0].keypoints  # Masks object

        black_frame = Extrac_Pose(keypoints,frame)
        
        cv2.imshow("Original Video", anot_frame)
        cv2.imshow("Extracted Pose", black_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()