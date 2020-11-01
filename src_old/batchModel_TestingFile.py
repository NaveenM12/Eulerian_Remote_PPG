import os
import numpy as np
import cv2
import time
import scipy.fftpack as fftpack
from scipy import signal

#from PIL import Image
import numpy as np

#from imutils import face_utils

#import dlib

#import multiprocessing as mp

from threading import Thread


import pyramids
import eulerian
import heartrate

#finishedFirstBatch
#frames
#fps



class batchModel(object):

    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.finishedFirstBatch = False
        self.windowSize = 50
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.findNextFrames, args=())
        self.thread.daemon = True
        self.thread.start()
        self.faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")


    def findNextFrames(self):
        self.video_frames = []
        face_rects = ()

        while True:
            ret, img = self.capture.read()
            if not ret:
                break
            #cv2.imshow('frame',img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            roi_frame = img


            if(len(self.video_frames) == self.windowSize):
                self.video_frames.pop(0)

            # Detect face

            face_rects = self.faceCascade.detectMultiScale(gray, 1.3, 5)

            # Select ROI
            if len(face_rects) > 0:
                for (x, y, w, h) in face_rects:
                    roi_frame = img[y:y + h, x:x + w]
                if roi_frame.size != img.size:
                    roi_frame = cv2.resize(roi_frame, (500, 500))
                    frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                    frame[:] = roi_frame * (1. / 255)
                    self.video_frames.append(frame)

                print(len(self.video_frames))

    def calcBPM(self):

        heart_rate = 0

        currframes = self.video_frames.copy()
        # Build Laplacian video pyramid
        print("Building Laplacian video pyramid...")
        lap_video = pyramids.build_video_pyramid(currframes)
        freq_min = 1
        freq_max = 1.8

        amplified_video_pyramid = []

        for i, video in enumerate(lap_video):
            if i == 0 or i == len(lap_video) - 1:
                continue

            #cv2.imshow("frame", video)
            # Eulerian magnification with temporal FFT filtering
            print("Running FFT and Eulerian magnification...")
            result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, self.fps)
            lap_video[i] += result

            # Calculate heart rate
            print("Calculating heart rate...")
            heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)
        print(heart_rate)


if __name__ == '__main__':
    batchModel = batchModel()
    time.sleep(10)
    while True:
        try:
            batchModel.calcBPM()
        except AttributeError:
            pass
