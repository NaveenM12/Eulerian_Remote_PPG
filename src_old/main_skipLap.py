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



faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")

#FPS = 12 #This just sets the output speed, but it's not capturing that fast...
#NUM_FRAMES = 120

# Frequency range for Fast-Fourier Transform
# freq_min = 0.4
# freq_max = 3
freq_min = 1
freq_max = 1.8



fourcc = cv2.VideoWriter_fourcc(*'XVID')

cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))

video_frames = []
face_rects = ()


first = True
numFrames = 0;
frames = [];

while True:

    # End when q is entered by user
    k = cv2.waitKey(1) & 0xFF
    # press 'q' to exit
    if k == ord('q'):
        break


    # Read in frame
    correct, frame = cap.read()
    if not correct:
        break

    # We will now have to detect the face and extract the region of interest from the image
    # First, convert the frame to grayscale for this part
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # For this method, we will be using two ROIs, each cheek
    #ROI1 = np.zeros((10, 10, 3), np.uint8)
    #ROI2 = np.zeros((10, 10, 3), np.uint8)

    face_frame = frame


    # Detects the face appearences using cascade and creates a box around the face
    face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Select ROI
    if len(face_rects) > 0:
        for (x, y, w, h) in face_rects:
            face_frame = frame[y:y + h, x:x + w]

        # If the region of interest has been subsetted (face has been detected, then we will continue)
        if face_frame.size != frame.size:

            # Resize the face
            face_frame = cv2.resize(face_frame, (500, 500))

            # rect is the face detected
            shape = predictor(gray_img, rect)
            shape = face_utils.shape_to_np(shape)

            roi_1 = face_frame[shape[29][1]:shape[33][1], shape[54][0]:shape[12][0]]  # right cheeks
            roi_2 = face_frame[shape[29][1]:shape[33][1], shape[4][0]:shape[48][0]]  # left cheek


            print("eulerification...")


            fft = fftpack.fft(face_frame, axis=0)
            frequencies = fftpack.fftfreq(face_frame.shape[0], d=1.0 / fps)
            #frequencies = fftpack.fftfreq(len(lap_video[0]), d=1.0 / fps)
            bound_low = (np.abs(frequencies - freq_min)).argmin()
            bound_high = (np.abs(frequencies - freq_max)).argmin()
            fft[:bound_low] = 0
            fft[bound_high:-bound_high] = 0
            fft[-bound_low:] = 0
            iff = fftpack.ifft(fft, axis=0)
            result = np.abs(iff)
            result *= 100  # Amplification factor

            #gauss_frame = np.add(gauss_frame, result, out= gauss_frame, casting="unsafe")


            # lap_video = pyramids.build_video_pyramid(face_frame)
            # result, fft, frequencies = eulerian.fft_filter(face_frame, freq_min, freq_max, fps)
            #
            # lap_video += result

            #cv2.imshow('frame', face_frame)
            #print(result)

            #img = Image.fromarray(result, 'RGB')
            #img.show()

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
            #
            # pyr_up_frame = cv2.pyrUp(lap_video)
            # (height, width, depth) = pyr_up_frame.shape
            # prev_level_frame = face_frame
            # prev_level_frame = cv2.resize(face_frame, (height, width))
            # prev_frame = pyr_up_frame + prev_level_frame
            #
            # # Normalize pixel values
            # min_val = min(0.0, prev_frame.min())
            # prev_frame = prev_frame + min_val
            # max_val = max(1.0, prev_frame.max())
            # prev_frame = prev_frame / max_val
            # prev_frame = prev_frame * 255
            #
            # prev_frame = cv2.convertScaleAbs(prev_frame)
            #
            # cv2.imshow('f', prev_frame)


print ("Estimated frames per second : {0}".format(fps));
cap.release()
cv2.destroyAllWindows()