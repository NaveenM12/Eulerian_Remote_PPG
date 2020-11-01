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


    def findNextFrames(self):

        self.video_frames = []

        faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")

        face_rects = ()

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')

        while True:

            valid, frame = self.capture.read()

            if not valid:
                break

            # We will now have to detect the face and extract the region of interest from the image
            # First, convert the frame to grayscale for this part
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            roi_frame = gray

            if(len(self.video_frames) == self.windowSize):
                self.video_frames.pop(0)
                self.finishedFirstBatch = True


            face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)
            #print("face detected")

            # Select ROI
            if len(face_rects) > 0:
                print("face detected")
                for (x, y, w, h) in face_rects:
                    roi_frame = frame[y:y + h, x:x + w]
                if roi_frame.size != frame.size:
                    roi_frame = cv2.resize(roi_frame, (500, 500))
                    frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                    frame[:] = roi_frame * (1. / 255)
                    self.video_frames.append(frame)
                    print("face added")
                    print(len(self.video_frames))



    def calcBPM(self):

        print("entered main thread...")

        waiting = True

        while not self.finishedFirstBatch:
            waiting = True

        heart_rate = 0

        # Build Laplacian video pyramid
        print("Building Laplacian video pyramid...")

        # Frequency range for Fast-Fourier Transform
        freq_min = 1
        freq_max = 1.8

        levels = 3

        laplacian_video = []

        currframes = self.video_frames.copy()

        for i, frame in enumerate(currframes):

            # Build the gaussian pyramid -----------------------------------------------------
            img = np.ndarray(shape=frame.shape, dtype="float")
            img[:] = frame
            gaussian_pyramid = [img]

            for count in range(levels - 1):
                img = cv2.pyrDown(img)
                gaussian_pyramid.append(img)
            # Gaussian Pyramid created ! ----------------------------------------------------


            # Now that we have our adjusted video, eulerify and find the heartbeats!
            for i, video in enumerate(gaussian_pyramid):
                if i == 0 or i == len(gaussian_pyramid) - 1:
                    continue

                # Eulerian magnification with temporal FFT filtering ------------------------------------------------
                print("eulerification...")

                fft = fftpack.fft(video, axis=0)
                frequencies = fftpack.fftfreq(video.shape[0], d=1.0 / self.fps)
                # frequencies = fftpack.fftfreq(len(lap_video[0]), d=1.0 / fps)
                bound_low = (np.abs(frequencies - freq_min)).argmin()
                bound_high = (np.abs(frequencies - freq_max)).argmin()
                fft[:bound_low] = 0
                fft[bound_high:-bound_high] = 0
                fft[-bound_low:] = 0
                iff = fftpack.ifft(fft, axis=0)
                result = np.abs(iff)
                result *= 100  # Amplification factor

                print("calculating bpm...")

                fft_maximums = []

                for i in range(fft.shape[0]):
                    if freq_min <= frequencies[i] <= freq_max:
                        fftMap = abs(fft[i])
                        fft_maximums.append(fftMap.max())
                    else:
                        fft_maximums.append(0)

                peaks, properties = signal.find_peaks(fft_maximums)
                max_peak = -1
                max_freq = 0

                # Find frequency with max amplitude in peaks
                for peak in peaks:
                    if fft_maximums[peak] > max_freq:
                        max_freq = fft_maximums[peak]
                        max_peak = peak

                print(frequencies[max_peak] * 60)

if __name__ == '__main__':
    batchModel = batchModel()
    time.sleep(1)
    while True:
        try:
            batchModel.calcBPM()
        except AttributeError:
            pass
