
import cv2
import math
import random
import colorsys
import numpy as np

from mrcnn.model import MaskRCNN
from mrcnn import visualize
from mrcnn import model as modellib
from mrcnn.config import Config

class SimpleConfig(Config):
    # give the configuration a recognizable name
    NAME = "coco_inference"

    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81
    

    # number of classes (we would normally add +1 for the background
    # but the background class is *already* included in the class names)

class MyDetector:
    def __init__(self,LABELS_FILE,WEIGHTS_FILE):
        self.LABELS_FILE = LABELS_FILE
        self.WEIGHTS_FILE = WEIGHTS_FILE

        self.IMAGE_RESIZE_FACTOR = 0.5
        self.MinDistance = 70  # MIN DISTANCE TO ZONE PIXELS
        self.Factor = 0.3  # ZONE FACTOR
        self.CLASS_NAMES = open(self.LABELS_FILE).read().strip().split("\n")

        hsv = [(i / len(self.CLASS_NAMES), 1, 1.0) for i in range(len(self.CLASS_NAMES))]
        self.COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

        config = SimpleConfig()
        config.NUM_CLASSES = len(self.CLASS_NAMES)

        print("[INFO] loading Mask R-CNN model...")
        self.model = MaskRCNN(
            mode="inference", config=config, model_dir='./model_logs')
        self.model.load_weights(self.WEIGHTS_FILE, by_name=True)

    def Detect(self, frame):
        DrawFrame=frame.copy()
        DrawFrame = cv2.cvtColor(DrawFrame, cv2.COLOR_BGR2RGB)
        self.results = self.model.detect([DrawFrame], verbose=1)[0]
        for i in range(0, self.results["rois"].shape[0]):
            # extract the class ID and mask for the current detection, then
            # grab the color to visualize the mask (in BGR format)
            classID = self.results["class_ids"][i]
            mask = self.results["masks"][:, :, i]
            color = self.COLORS[classID][::-1]

            # visualize the pixel-wise mask of the object
            DrawFrame = visualize.apply_mask(DrawFrame, mask, color, alpha=0.5)

        # convert the image back to BGR so we can use OpenCV's drawing
        # functions
        DrawFrame = cv2.cvtColor(DrawFrame, cv2.COLOR_RGB2BGR)

        # loop over the predicted scores and class labels
        for i in range(0, len(self.results["scores"])):
            # extract the bounding box information, class ID, label, predicted
            # probability, and visualization color
            (startY, startX, endY, endX) = self.results["rois"][i]
            classID = self.results["class_ids"][i]
            label = self.CLASS_NAMES[classID]
            score = self.results["scores"][i]
            color = [int(c) for c in np.array(self.COLORS[classID]) * 255]

            # draw the bounding box, class label, and score of the object
            cv2.rectangle(DrawFrame, (startX, startY), (endX, endY), color, 2)
            text = "{}: {:.3f}".format(label, score)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(DrawFrame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
        return DrawFrame

    def WatchIfCarOnZone(self, frame, ConfigJsonFile):
        Zones=ConfigJsonFile["Zones"]
        NumOfZones = len(Zones)
        DrawFrame = frame.copy()
        if len(self.results['scores']) == 0:
            return DrawFrame

        masks = self.results['masks']
        zip_thing = zip(
            self.results['scores'],
            self.results['class_ids'],
            self.results['rois'],
            range(0, len(masks[0, 0, :]))
        )
        for scores, clases_id, match, i in zip_thing:
            distanceToZone = np.zeros([4, NumOfZones])
            Contador = 0
            Index = -1

            for r in range(0, NumOfZones):
                Contador = 0
                pts3 = [Zones[r]["xinit"], Zones[r]["yinit"], Zones[r]["xend"], Zones[r]["yend"]]
                # Esquina superior izquierda
                dx1 = match[1] - pts3[0]
                dy1 = match[0] - pts3[1]
                # Esquina inferior derecha
                dx2 = match[3] - pts3[2]
                dy2 = match[2] - pts3[3]

                # Esquina superior derecha
                dx3 = dx2
                dy3 = dy1
                # Esquina inferior izq
                dx4 = dx1
                dy4 = dy2

                distanceToZone[0][r] = math.hypot(dy1, dx1)
                if(distanceToZone[0][r] < self.MinDistance):
                    Contador += 1

                distanceToZone[1][r] = math.hypot(dy2, dx2)
                if(distanceToZone[1][r] < self.MinDistance):
                    Contador += 1

                distanceToZone[2][r] = math.hypot(dy3, dx3)
                if(distanceToZone[2][r] < self.MinDistance):
                    Contador += 1

                distanceToZone[3][r] = math.hypot(dy4, dx4)
                if(distanceToZone[3][r] < self.MinDistance):
                    Contador += 1

                if(Contador == 4):
                    Index = r

                print("distanceToZone", distanceToZone)

            if Index != -1:
                print("!!!!!!!!!El auto ha [[ENTRADO]] A la ZONA #: ", Index)
                bbox = (
                    int(match[1] + (match[3] - match[1]) * self.Factor),
                    int(match[0] + (match[2] - match[0]) * self.Factor),
                    int((match[3] - match[1]) * (1 - 2 * self.Factor)),
                    int((match[2] - match[0]) * (1 - 2 * self.Factor))
                )
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(DrawFrame, p1, p2, (255, 0, 0), 2, 1)
            else:
                print("!!!!!!!!!El auto ha [[SALIDO]] de la ZONA #: ",Index)
        return DrawFrame