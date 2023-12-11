# Pose Detection and Extraction using YOLOv8

![git](https://github.com/abdullah1772/Yolo-Pose-Extraction/assets/88187437/ec963487-85ec-4913-b601-df2946e932ef)

This repository contains Python scripts for extracting human poses from videos using the YOLO (You Only Look Once) object detection model. The project consists of two main scripts: pose.py and pose_extract.py.

## pose.py

This script uses the YOLO model for detecting human figures in a video and extracts their poses. It reads a video file, processes each frame to detect human figures, and then extracts their pose keypoints.

## Dependencies

os
cv2 (OpenCV)
numpy
pandas
matplotlib
ultralytics (YOLO)

## Usage

To use this script, ensure you have a video file in the specified directory and run the script. The output will be the video frames with detected human poses.

## pose_extract.py

This script contains functions to assist in the extraction and processing of pose keypoints from the detected human figures.

## Dependencies
numpy
matplotlib
cv2 (OpenCV)

## Key Point Definitions

The script includes definitions for various key points in the human body such as the nose, eyes, shoulders, elbows, hands, hips, knees, and feet.

## Usage

This is a helper script used by pose.py and is not intended to be run independently.

## Example

Provide an example of how to run the script, such as:

python pose.py
