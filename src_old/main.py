import os
import numpy as np
import cv2
import time
import scipy.fftpack as fftpack
from scipy import signal

from PIL import Image
import numpy as np



faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")

#FPS = 12 #This just sets the output speed, but it's not capturing that fast...
#NUM_FRAMES = 120

# Frequency range for Fast-Fourier Transform
freq_min = 1
freq_max = 1.8

fourcc = cv2.VideoWriter_fourcc(*'XVID')

cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))

video_frames = []
face_rects = ()

while True:

    k = cv2.waitKey(1) & 0xFF
    # press 'q' to exit
    if k == ord('q'):
        break

    ret, img = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    roi_frame = img


    # Detect face
    if len(video_frames) == 0:
        face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)

    # Select ROI
    if len(face_rects) > 0:
        for (x, y, w, h) in face_rects:
            roi_frame = img[y:y + h, x:x + w]
        if roi_frame.size != img.size:
            roi_frame = cv2.resize(roi_frame, (500, 500))


            fft = fftpack.fft(roi_frame, axis=0)
            frequencies = fftpack.fftfreq(roi_frame.shape[0], d=1.0 / fps)
            bound_low = (np.abs(frequencies - freq_min)).argmin()
            bound_high = (np.abs(frequencies - freq_max)).argmin()
            fft[:bound_low] = 0
            fft[bound_high:-bound_high] = 0
            fft[-bound_low:] = 0
            iff = fftpack.ifft(fft, axis=0)
            result = np.abs(iff)
            result *= 100  # Amplification factor



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


print ("Estimated frames per second : {0}".format(fps));
cap.release()
cv2.destroyAllWindows()