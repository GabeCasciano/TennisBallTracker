## Tutorial -> "https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/"
import numpy as np
import cv2
from imutils.video import VideoStream
import imutils
import time
import argparse
import platform

lower = (23, 75, 75)
upper = (65, 255, 255)

def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=640, display_height=480, framerate=60, flip_method=2):
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
        print("Nope")

    #Main loop
    while True:
        #get stream image
        ret, frame = vs.read()
        #if the image is empty break and re-try
        if frame is None:
            print("Nope 2")
            break

        frame = imutils.resize(frame, width=600, height=400) # resize the image
        blurred = cv2.GaussianBlur(frame, (9,9), 0) # apply a Gaussian blur
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) # convert to HSV color space

        kernel = np.ones((2,2), np.uint8)
        mask = cv2.inRange(hsv, lower, upper) # Selecting color from image based on bounds

        diliated = cv2.dilate(mask, kernel, iterations=2) #
        erroded = cv2.erode(diliated, kernel, iterations=1)  # Erode the image

        # Find the contours on the mask and find the current center of the ball
        contours = cv2.findContours(erroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        center = None

        if len(contours)>0:

            c = max(contours, key=cv2.contourArea)
            ((x,y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"]/M["m00"]))
           # print("Ball found at " + center)
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0,0,255), -1)

        cv2.imshow("Ball", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Erroded", erroded)
        cv2.imshow("Dialated", diliated)
        cv2.imshow("blurr", blurred)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quit")
            break

    vs.stop()
    vs.release()
    cv2.destroyAllWindows()
    return

main_loop() # run main
