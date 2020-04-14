import numpy as np
import cv2
import imutils
import time
import argparse

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=2):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
    """
    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){capture_width}, height=(int){capture_height}, ' +
            f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
            f'nvvidconv flip-method={flip_method} ! ' +
            f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )

def running_on_jetson_nano():
    # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
    # so that we can access the camera correctly in that case.
    # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
    return platform.machine() == "aarch64"

def main_loop():
    #Check where this is running
    if running_on_jetson_nano():
        vs = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
    else:
        vs = cv2.VideoCapture(0)
    #Main loop
    while True:
        #get stream image
        ret, frame = vs.read()

        #if the image is empty break and re-try
        if frame is None:
            break

        frame = imutils.resize(frame, width=600) # resize the image
        blurred = cv2.GaussianBlur(frame, (11,11), 0) # apply a Gaussian blur
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) # convert to HSV color space

        # Tutorial -> "https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/"


    return

