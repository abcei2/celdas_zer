import cv2
import colorsys
import numpy as np

from mrcnn.model import MaskRCNN

from functions import *
from ZoneAnalizer import *


VIDEO_SOURCE = "http://199.48.198.27/mjpg/video.mjpg"

LABELS_FILE = 'coco_labels.txt'
WEIGHTS_FILE = 'mask_rcnn_coco.h5'

IMAGE_RESIZE_FACTOR = 0.5
MinDistance = 70  # MIN DISTANCE TO ZONE PIXELS
Factor = 0.3  # ZONE FACTOR

InitiZones = False
#####################################

cap = cv2.VideoCapture(VIDEO_SOURCE)

while not InitiZones:
    ret, frame = cap.read()
    frame = resize_frame(frame, IMAGE_RESIZE_FACTOR)
    cv2.imshow("image", frame[:, :, :])
    aux = cv2.waitKey(100)
    if(aux == 97):  # a
        print(InitiZones)
        InitiZones = True


# Read first frame.
ret, frame = cap.read()
frame = resize_frame(frame, IMAGE_RESIZE_FACTOR)
DrawFrame = frame.copy()
Zone, pts = GetZones(DrawFrame)


# load the class label names from disk, one label per line
CLASS_NAMES = open(LABELS_FILE).read().strip().split("\n")

hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

config = SimpleConfig()
config.NUM_CLASSES = len(CLASS_NAMES)

print("[INFO] loading Mask R-CNN model...")
model = MaskRCNN(mode="inference", config=config, model_dir='./model_logs')
model.load_weights(WEIGHTS_FILE, by_name=True)

while True:

    ret, frame = cap.read()
    frame = resize_frame(frame, IMAGE_RESIZE_FACTOR)
    DrawFrame = frame.copy()

    # VISUALICE ZONES ON IMAGE. CAN BE REMOVE
    for i, point in enumerate(pts):
        cv2.rectangle(DrawFrame, point[0], point[1], (0, 255, 0), 2, 1)

    ########
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.detect([image], verbose=1)[0]
    WatchIfCarOnZone(results, frame, DrawFrame, MinDistance, Factor, Zone, pts)

    for i in range(0, len(results["scores"])):
        (startY, startX, endY, endX) = results["rois"][i]
        classID = results["class_ids"][i]
        label = CLASS_NAMES[classID]
        score = results["scores"][i]
        color = [int(c) for c in np.array(COLORS[classID]) * 255]
        cv2.rectangle(DrawFrame, (startX, startY), (endX, endY), color, 2)

    print("\n\n#######################################\n\n")
    cv2.imshow("image", DrawFrame)
    cv2.waitKey(10)
