import os
import numpy as np
import cv2
import time
import math
import scipy
import scipy.fftpack as fftpack
from scipy import signal
import imutils
from imutils import face_utils
from statistics import mode
import numpy as np
#from PIL import Image
#from imutils import face_utils
#import dlib
#import multiprocessing as mp
from threading import Thread


class threadModel(object):

    # Initialize the inputs for the modoel, starting the input thread and setting key input variables
    def __init__(self, src=0):
        # our src will be the webcam
        self.capture = cv2.VideoCapture(src)
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.finishedFirstBatch = False

        # The windoow size indicates that we will be evaluating 100 frames at a time, continually updating with
        # the newest frames
        self.windowSize = 75
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.findNextFrames, args=())
        self.thread.daemon = True
        self.thread.start()


    # This method will continuously read in input in its own thread
    def findNextFrames(self):

        self.video_frames = []

        faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")

        while True:

            # Too be used foor coordinates oof face rectangles
            face_rects = ()

            # read and verify data
            valid, frame = self.capture.read()
            if not valid:
                break

            # We will now have to detect the face and extract the region of interest from the image
            # First, convert the frame to grayscale for this part
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            roi_frame = frame

            # detect face coordinates with Cascade
            face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)
            #print("face detected")

            # Select Region of Interest if available, then extract it from the image and update the list of current
            # frames
            if len(face_rects) > 0:
                #print("face detected")
                for (x, y, w, h) in face_rects:
                    roi_frame = frame[y:y + h, x:x + w]
                if roi_frame.size != frame.size:

                    #UNCOMMENT THE CODE BELOW TO RUN THE PROGRAM ONLY ON THE CHEEKS AND FOREHEAD, NOT THE ENTIRE FACE!
                    '''
                    width_scaled, height_scaled = roi_frame.shape[1] * 0.40, roi_frame.shape[0] * 0.40
                    center_x, center_y = roi_frame.shape[1] / 2, roi_frame.shape[0] / 2
                    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
                    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
                    right_cheek = roi_frame[int(top_y)+85:int(bottom_y), int(left_x)+146:int(right_x)+30].copy()

                    left_cheek = roi_frame[int(top_y)+90:int(bottom_y), int(left_x)-25:int(right_x)-160].copy()


                    width_scaled, height_scaled = roi_frame.shape[1] * 0.50, roi_frame.shape[0] * 0.50
                    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
                    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
                    forehead = roi_frame[int(top_y)-85:int(bottom_y - 235), int(left_x):int(right_x)].copy()

                    if right_cheek.size == 0 or left_cheek.size == 0 or forehead.size == 0:
                        #print("face partially coovered, moving on!")
                        continue

                    right_cheek = imutils.resize(right_cheek, width=800)
                    left_cheek = imutils.resize(left_cheek, width=800)
                    forehead = imutils.resize(forehead, width=800)

                    roi_frame = right_cheek

                    roi_frame = np.concatenate((right_cheek, left_cheek), axis=0)
                    roi_frame = np.concatenate((roi_frame, forehead), axis=0)
                    '''


                    roi_frame = cv2.resize(roi_frame, (500, 500))
                    frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                    frame[:] = roi_frame * (1.0 / 255)

                    # If we are already hoolding the max number of frames we want too analyze, we will have to get
                    # rid fo the first frame
                    if (len(self.video_frames) >= self.windowSize):
                        self.video_frames.pop(0)
                        self.finishedFirstBatch = True

                    # Adds the current ROI to the frame list
                    self.video_frames.append(frame)

                    #print(len(self.video_frames))
                    #print("face added")
                    #print(len(self.video_frames))
            #else:
                #print("no face detected")


    # This method will calculate and print the BPM foor the current list of frames
    def calcBPM(self):

       # print("entered main thread...")

        # If the minimum number of frames haven't been read, then wait
        while not self.finishedFirstBatch:
            time.sleep(3)
            t = 3
            print("waiting for first batch....")

        '''
        We will now build the laplacian (enhanced) video, which creates a pyramid of images to identify changes 
        between frames
        '''
        # Frequency range for Fast-Fourier Transform (used to detect heartbeat)
        freq_min = 1
        freq_max = 1.8

        # How many levels we want our pyramid to have
        levels = 3

        self.laplacian_video = []
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

            # Build the laplacian pyramid and video ------------------------------------------
            laplacian_pyramid = []

            for k in range(levels - 1):
                u = cv2.pyrUp(gaussian_pyramid[k + 1])
                (height, width, depth) = u.shape
                gaussian_pyramid[k] = cv2.resize(gaussian_pyramid[k], (height, width))
                diff = cv2.subtract(gaussian_pyramid[k], u)
                laplacian_pyramid.append(diff)

            laplacian_pyramid.append(gaussian_pyramid[-1])

            for j in range(levels):
                if i == 0:
                    self.laplacian_video.append(
                        np.zeros((len(currframes), laplacian_pyramid[j].shape[0], laplacian_pyramid[j].shape[1], 3)))
                self.laplacian_video[j][i] = laplacian_pyramid[j]
            # Laplacian video and pyramid created! -----------------------------------


       # Now that we have our adjusted video, apply magnification and find the heartbeats!
        frequencies = []
        for i, video in enumerate(self.laplacian_video):
            if i == 0 or i == len(self.laplacian_video) - 1:
                continue

            # Eulerian magnification with temporal FFT filtering -----------------------
            #print("eulerification...")
            fft = fftpack.fft(video, axis=0)
            frequencies = fftpack.fftfreq(video.shape[i], d=1.0 / self.fps)
            # print(self.fps)
            bound_low = (np.abs(frequencies - freq_min)).argmin()
            bound_high = (np.abs(frequencies - freq_max)).argmin()
            fft[:bound_low] = 0
            fft[bound_high:-bound_high] = 0
            fft[-bound_low:] = 0
            iff = fftpack.ifft(fft, axis=0)
            result = np.abs(iff)
            result *= 100  # Amplification factor

            self.laplacian_video[i] += result


            # Heartrate Calculation ---------------------------------------------------------
            #print("calculating bpm...")

            fft_maximums = []
            fftMap = []


            for i in range(fft.shape[0]):
                if freq_min <= frequencies[i] <= freq_max:
                    fftMap = abs(fft[i])
                    fft_maximums.append(fftMap.max())
                else:
                    fft_maximums.append(0)

            peaks, properties = signal.find_peaks(fft_maximums)
            max_peak = -1
            max_freq = 0
            #sum = 0
            #holder = []

            # Find frequency with max amplitude in peaks
            for peak in peaks:
                #sum += fft_maximums[peak]
                #holder.append(fft_maximums[peak])
                if fft_maximums[peak] > max_freq:
                    max_freq = fft_maximums[peak]
                    max_peak = peak

        print("BPM: " + str(round(frequencies[max_peak] * 60, 2)))

        # Create a thread to display the enhanced video, while the next frames can be calculated
        self.display = Thread(target=self.displayEnhanced(), args=())
        self.display.daemon = True
        self.display.start()

    # displays the enhanced video through collapsingg the Laplacian pyram
    def displayEnhanced(self):
        # Display the enhanced Video -----------------------------------------------------------
        collapsed_video = []

        for i in range(self.windowSize-1):
            prev_frame = self.laplacian_video[-1][i]

            for level in range(len(self.laplacian_video) - 1, 0, -1):
                pyr_up_frame = cv2.pyrUp(prev_frame)
                (height, width, depth) = pyr_up_frame.shape
                prev_level_frame = self.laplacian_video[level - 1][i]
                prev_level_frame = cv2.resize(prev_level_frame, (height, width))
                prev_frame = pyr_up_frame + prev_level_frame

            # Normalize pixel values
            min_val = min(0.0, prev_frame.min())
            prev_frame = prev_frame + min_val
            max_val = max(1.0, prev_frame.max())
            prev_frame = prev_frame / max_val
            prev_frame = prev_frame * 255

            prev_frame = cv2.convertScaleAbs(prev_frame)
            collapsed_video.append(prev_frame)

        for frame in collapsed_video:
            cv2.imshow("frame", frame)
            cv2.waitKey(7)


# Start both the threads
if __name__ == '__main__':
    # initialize the first thread for data collection
    threadModel = threadModel()
    time.sleep(1)
    while True:
        try:
            # continuously analyze the data stream
            threadModel.calcBPM()
        except (AttributeError, KeyboardInterrupt) as e:
            pass
