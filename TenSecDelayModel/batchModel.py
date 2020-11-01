import os
import numpy as np
import cv2
import time
import scipy.fftpack as fftpack
from scipy import signal

from PIL import Image
import numpy as np

from imutils import face_utils

import dlib

import multiprocessing as mp

def getNext():
    findNextFrames()

class batchModel:


    def findNextFrames():

        global finishedFirstBatch
        finishedFirstBatch = False

        global frames
        frames = []

        cap = cv2.VideoCapture(0)

        frameCount = 0;

        firstInSet = True

        framesDesired = 20

        face_rects = ()

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # fps = int(cap.get(cv2.CAP_PROP_FPS))

        while True:

            valid, frame = cap.read()

            if not valid:
                break
            else:
                print("reading frames")
                cv2.imshow('frame', frame)


            # We will now have to detect the face and extract the region of interest from the image
            # First, convert the frame to grayscale for this part
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)


            # Select ROI
            if len(face_rects) > 0:
                for (x, y, w, h) in face_rects:
                    roi_frame = frame[y:y + h, x:x + w]
                if roi_frame.size != frame.size:
                    roi_frame = cv2.resize(roi_frame, (500, 500))
                    frames[frameCount].append(roi_frame)

                    if frameCount >= framesDesired:
                        frames[0].pop()
                        finishedFirstBatch = True
                    else:
                        frameCount += 1


    def main(self):

        global faceCascade
        faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")



        # Frequency range for Fast-Fourier Transform
        # freq_min = 0.4
        # freq_max = 3
        freq_min = 1
        freq_max = 1.8



        first = True
        levels = 3

        finishedFirstBatch = False


        while not finishedFirstBatch:
            # print("Waiting for First Batch")
            nn = 13

        while True:

            # End when q is entered by user
            k = cv2.waitKey(1) & 0xFF
            # press 'q' to exit
            if k == ord('q'):
                break

            laplacian_video = []

            currFrames = frames.copy()

            for i, frame in enumerate(currFrames):

                # Build the  pyramid -----------------------------------------------------------------------
                img = np.ndarray(shape=currFrames.shape, dtype="float")
                img[:] = currFrames.copy()
                gaussian_pyramid = [img]

                for count in range(levels - 1):
                    img = cv2.pyrDown(img)
                    gaussian_pyramid.append(img)

                laplacian_pyramid = []

                for k in range(levels - 1):
                    u = cv2.pyrUp(gaussian_pyramid[k + 1])
                    (height, width, depth) = u.shape
                    gaussian_pyramid[k] = cv2.resize(gaussian_pyramid[i], (height, width))
                    diff = cv2.subtract(gaussian_pyramid[k], u)
                    laplacian_pyramid.append(diff)

                laplacian_pyramid.append(gaussian_pyramid[-1])

                for j in range(3):
                    if i == 0:
                        laplacian_video.append(
                            np.zeros((len(frames), laplacian_pyramid[j].shape[0], laplacian_pyramid[j].shape[1], 3)))
                    laplacian_video[j][i] = laplacian_pyramid[j]
                # -----------------------------------------------------------------------------------------------

            for i, video in enumerate(laplacian_video):
                if i == 0 or i == len(laplacian_video) - 1:
                    continue

                # Eulerian magnification with temporal FFT filtering ------------------------------------------------

                print("eulerification...")

                fft = fftpack.fft(video, axis=0)
                frequencies = fftpack.fftfreq(video.shape[0], d=1.0 / fps)
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


    if __name__ == "__main__":

        mp.freeze_support()

        mp.set_start_method('forkserver',force=True)
        p = mp.Process(target=getNext, args=(self,))
        p.start()
        time.sleep(10)
        main()
        p.join()

    #print ("Estimated frames per second : {0}".format(fps));
    #cap.release()
   # cv2.destroyAllWindows()