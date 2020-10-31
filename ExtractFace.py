import os
import cv2
import numpy as np


def extractFace(img):
    # Define paths
    base_dir = os.path.dirname(__file__)
    prototxt_path = os.path.join(base_dir + 'extract_model/deploy.prototxt')
    caffemodel_path = os.path.join(base_dir + 'extract_model/weights.caffemodel')

    # Read the model
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


    # Create directory 'updated_images' if it does not exist
    if not os.path.exists('updated_images'):
        print("New directory created")
        os.makedirs('updated_images')


    # Create directory 'face' if it does not exist
    if not os.path.exists('faces'):
        print("New directory created")
        os.makedirs('face')


    image = cv2.imread(base_dir + 'images/' + file)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()




