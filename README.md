# Remote-PPG

## Implementation
The final implementation of the model is in the RemotePPG_src_final. This model uses the haarcascades library to locate and extract the face from the webcam input using a base ML model and then builds the eulerian and laplace pyramids to enhance the video, from which an ML model can then detect the heart rate. 
<br><br>
There are two key aspects of the model: face detection and magnification. The face detection through the camera to locate the user then magnification to identify the heart rate. The magnification works as shown below:
![](Eulerian_Magnification_Example.png)

## Building Off of Past Research/Models
This is a novel implementation of previous repositories as it allows for a real-time detection. The model will hold a given number of frames, then run the enhancement of that snippet of video and perform the BPM calculation on the current X number of frames. These frames are coontinuously updated until the program is stopped by the user, thus allowing for a real-time continuous implementation. This program uses a multi-threaded approach inorder to read in the input frames from the webcam, while also performing BPM/enhancement calculations**

## How to Run
The main code is in one file, 'BPM_Detector.py'. When run, the program will continuously read-in, process, and print the Beats Per Minute of the detected face. Up to this point, I've focused mainly on enhancing frames to count BPM (as opposed to other areas such as robust face recognition, graphical displays, etc.). My work was derived from this MIT project: http://people.csail.mit.edu/mrub/vidmag/ on Eulerian Video Magnification. The program will take around 30 seconds to process the initial batch of frames, but then will print out new BPM values every few seconds. Make sure your head is facing and close to the camera, if the program is unable to detect your face, it will just continuously print BPM results from old frames. There is also the option to run the program using data only from your cheeks & forehead (as opposed to your entire face), which can be done through commenting out a single block of code (Line 74). The BPM returned by the program tends to be fairly erratic; it always returns a range that contains the true value, but sometimes outliers cause this range to be around 40 BPM. I am thus still working on improving this by taking averages, finding mode values, or even using different libraries on top of the video magnification. The program does seem to get more accurate after it has made many guesses but still has lots of room for improvement. Also, as a side note: when using the forehead/cheek mode, the user's face has to be especially aligned and close to the webcam, and the model does not perform well when a face is directly in front of a light source as the face gets blacked out. 



### The other folders in this repo contain various PPG implementations of the same MIT paper linked below, from which the RemotePPG_src_final is based upon.
