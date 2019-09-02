import cv2
from mrcnn.config import Config


################
## CREATE TRACKER##
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(major_ver)

def CreateTracker(tracker_type):

    if int(major_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
    return tracker
################
## INIT ZONES##
global pts, clicked, Zone, DrawEscene
clicked = False
pts = [[(3, 1), (4, 2)]]
Zone = len(pts)

def click_and_crop(event, x, y, flags, param):
    # grab references to the glo
    global pts, clicked, Zone, DrawEscene
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        if ~clicked:
            pts[Zone-1][0] = (x, y)
            clicked = True
    elif event == cv2.EVENT_LBUTTONUP:
        pts[Zone-1][1] = (x, y)
        clicked = False
    elif event == 3:
        Zone = Zone+1
        pts.append([(10, 5), (20, 30)])
    elif event == cv2.EVENT_MOUSEMOVE:
        if(clicked):
            AuxFrame = DrawEscene.copy()
            cv2.rectangle(AuxFrame, pts[Zone-1][0], (x, y), (0, 255, 0),	1)
            cv2.imshow("image", AuxFrame)
            
def GetZones(Escene):
    global pts,  Zone, DrawEscene
    DrawEscene=Escene
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)    
    cv2.imshow("image", Escene)
    cv2.waitKey(0)
    return Zone, pts

############################
##MODEL CONFIG
class SimpleConfig(Config):
    # give the configuration a recognizable name
    NAME = "coco_inference"

    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES=81
