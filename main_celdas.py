
from mrcnn import model as modellib
import argparse
import os
import time
from functions import *
from ZoneAnalizer import *
# %matplotlib inline
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

###################################
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
                help="path to Mask R-CNN model weights pre-trained on COCO")
ap.add_argument("-l", "--labels", required=True,
                help="path to class labels file")
ap.add_argument("-i", "--image", required=True,
                help="path to input image to apply Mask R-CNN to")
args = vars(ap.parse_args())


cap = cv2.VideoCapture(
    "/home/pdi/Videos/prueba1cut.mp4")

######################################################################################
######################################TRACKING #######################################
######################################################################################
# global flagTrackers, trackers, DrawFrame

trackers = []
flagTrackers = []
MinDistance = 48  # MIN DISTANCE TO ZONE PIXELS
Factor = 0.45  # ZONE FACTOR

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]
######################################################################################
######################################################################################


InitiZones = False
#####################################
while(InitiZones == False):

    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("image", frame[:, :, :])
    aux = cv2.waitKey(100)
    if(aux == 97):  # a
        print(InitiZones)
        InitiZones = True
time.sleep(5)
FrameSkiping = 1
before = 0

# Read first frame.
ret, frame = cap.read()
DrawFrame = frame.copy()
Zone, pts = GetZones(DrawFrame)

for i in range(0, Zone):
    trackers.append(CreateTracker(tracker_type))
    flagTrackers.append(False)


# load the class label names from disk, one label per line
CLASS_NAMES = open(args["labels"]).read().strip().split("\n")

# initialize the inference configuration
config = SimpleConfig()
# number of classes (we would normally add +1 for the background
# but the background class is *already* included in the class
# names)
config.NUM_CLASSES = len(CLASS_NAMES)
# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)

while(True):

    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

    ########
    # VISUALICE ZONES ON IMAGE. CAN BE REMOVE
    DrawFrame = frame.copy()
    ########
    FrameSkiping = FrameSkiping+1

    if(FrameSkiping % 10 == 0):

        before = time.time()
        results = model.detect([frame], verbose=1)
        r = results[0]

        ########
        # VISUALICE ZONES ON IMAGE. CAN BE REMOVE
        cnt = 0
        for pts3 in pts:
            cv2.putText(DrawFrame, "Zona : "+str(cnt),
                        pts3[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cnt = cnt+1
        ########

        AnalizeOcupiedZone(trackers, flagTrackers, frame,
                           tracker_type, DrawFrame)

        WatchIfCarOnZone(r, trackers, flagTrackers, frame,
                         DrawFrame, MinDistance, Factor, Zone, pts)

        print("\n\n#######################################\n\n")
        cv2.imshow("image", DrawFrame[:, :, :])
        cv2.waitKey(10)

