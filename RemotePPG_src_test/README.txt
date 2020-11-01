This is the final implementation of the moodel. This model uses the haarcascades library to locate and extract the face from the webcam input using a base
ML model.

This program then builds the eulerian and laplace pyramids to enhance the video, from which an ML model can then detect the heart rate. 

** This is a novel implementation of previous repossitories as it allows for a real-time detection. The model will hold a given number of frames, then run
the enhancement of that snippet of video and perform the BPM calculation on the current X number of frames. These frames are coontinuously updated until the program
is stopped by the user, thus allowing for a real-time continuous implementation. **
