import os
import cv2
import random
import colorsys
import numpy as np

from mrcnn import visualize
from mrcnn import model as modellib
from mrcnn.config import Config


VIDEO_SOURCE = "http://199.48.198.27/mjpg/video.mjpg"

LABELS_FILE = 'coco_labels.txt'
WEIGHTS_FILE = 'mask_rcnn_coco.h5'

CLASS_NAMES = open(LABELS_FILE).read().strip().split("\n")


# generate random (but visually distinct) colors for each class label
# (thanks to Matterport Mask R-CNN for the method!)
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)


class SimpleConfig(Config):
    # give the configuration a recognizable name
    NAME = "coco_inference"

    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # number of classes (we would normally add +1 for the background
    # but the background class is *already* included in the class names)
    NUM_CLASSES = len(CLASS_NAMES)


# initialize the inference configuration
config = SimpleConfig()

# initialize the Mask R-CNN model for inference and then load the weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=os.getcwd())
print(os.getcwd())
model.load_weights(WEIGHTS_FILE, by_name=True)

# load the input image, convert it from BGR to RGB channel
# ordering, and resize the image
cap = cv2.VideoCapture(VIDEO_SOURCE)
FrameSkiping = 1

while True:

    _, image = cap.read()
    image = cv2.resize(
        image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = imutils.resize(image, width=512)

    FrameSkiping = FrameSkiping + 1

    if(FrameSkiping % 30 == 0):
        # perform a forward pass of the network to obtain the results
        print("[INFO] making predictions with Mask R-CNN...")
        r = model.detect([image], verbose=1)[0]

        # loop over of the detected object's bounding boxes and masks
        for i in range(0, r["rois"].shape[0]):
            # extract the class ID and mask for the current detection, then
            # grab the color to visualize the mask (in BGR format)
            classID = r["class_ids"][i]
            mask = r["masks"][:, :, i]
            color = COLORS[classID][::-1]

            # visualize the pixel-wise mask of the object
            image = visualize.apply_mask(image, mask, color, alpha=0.5)

        # convert the image back to BGR so we can use OpenCV's drawing
        # functions
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # loop over the predicted scores and class labels
        for i in range(0, len(r["scores"])):
            # extract the bounding box information, class ID, label, predicted
            # probability, and visualization color
            (startY, startX, endY, endX) = r["rois"][i]
            classID = r["class_ids"][i]
            label = CLASS_NAMES[classID]
            score = r["scores"][i]
            color = [int(c) for c in np.array(COLORS[classID]) * 255]

            # draw the bounding box, class label, and score of the object
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            text = "{}: {:.3f}".format(label, score)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(10)
