# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import cv2
import time
import imutils
import argparse
import numpy as np
from imutils.video import VideoStream
from centroidtracker import CentroidTracker


# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prototxt", required=True, type=str, help="path to Caffe deploy prototxt file")
parser.add_argument("-m", "--model", required=True, type=str, help="path to Caffe pre-trained model")
parser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

args = vars(parser.parse_args())
blue = (255,165,0)
# initialize our centroid tracker and frame dimensions
centroid_tracker = CentroidTracker()
(Height, Width) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
video_stream = VideoStream(src=0).start()
time.sleep(2.0)

# loop through every frame of the video stream
while True:
    # read the next frame from the video stream and resize it
    frame = video_stream.read()
    frame = imutils.resize(frame, width=400)

    if Height is None or Width is None:
        # get the first two dimensions of the frame
        (Height, Width) = frame.shape[:2]

    # construct blob from the frame
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(Width, Height), mean=(104.0, 177.0, 123.0))
    # default for swapping Red and Blue channel (swapRB) is false

    # pass the blob through the model and get the predictions
    net.setInput(blob)
    detections = net.forward()
    rects = list()  # bounding box list

    # loop over detection
    for i in range(detections.shape[2]):
        # select only prediction with probability larger than a threshold
        if detections[0, 0, i, 2] > 0.5:  # args["confidence"]:
            # get all the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([Width, Height, Width, Height])
            rects.append(box.astype("int"))

            # draw bounding boxes based on their coordinates:
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          color=blue, thickness=2)

    # with the list of bounding boxes, update centroid tracker
    objects = centroid_tracker.update(rects)

    # display ID number with all objects detected and tracked:
    for (objectID, centroid) in objects.items():
        text_display = "ID {}".format(objectID)
        cv2.putText(frame, text=text_display, org=(centroid[0] - 10, centroid[1] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=blue, thickness=3)
        cv2.circle(frame, centroid, radius=5, color=blue, thickness=-1)

    cv2.imshow("Face tracking", frame)

    # set up an exit key, display the frame until a designated key is pressed
    key = cv2.waitKey(1) & 0xFF

    # The 0xFF is added to ensure that even when NumLock is activated, key pressed will have the same result
    # For more explanation, look here: https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1

    # break the loop after 1s when q(quit) is pressed
    if key == ord('q'):
        break

# do a bit of cleanup
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
video_stream.stop()
