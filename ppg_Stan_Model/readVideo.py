import os
import numpy as np
import cv2
import time



#filename = '../video/video.avi'
FPS = 12 #This just sets the output speed, but it's not capturing that fast...
NUM_FRAMES = 120

cap = cv2.VideoCapture(0)

## some videowriter props
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*'XVID')

## open and set props
vout = cv2.VideoWriter()
vout.open('output.avi',fourcc,FPS,size,True)


start = time.time()
for i in range(NUM_FRAMES) :
    print(i);
    ret, frame = cap.read()
    if ret == True:
        vout.write(frame)    #TODO: I think this write takes too long...only getting about 12 FPS
        #cv2.imshow('frame', frame)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break;

    else:
        print('Error...')
        break;

end = time.time()
seconds = end - start
print("Time taken : {0} seconds".format(seconds))
fps  = NUM_FRAMES / seconds;
print ("Estimated frames per second : {0}".format(fps));
cap.release()
vout.release()
cv2.destroyAllWindows()