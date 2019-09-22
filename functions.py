import cv2
from mrcnn.config import Config


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
    elif event == 2:
        Zone = Zone+1
        pts.append([(10, 5), (20, 30)])
        print("Points", pts)
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


def resize_frame(frame, factor):
    return cv2.resize(
        frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
    )
