import numpy as np   
import math
import cv2
from functions import CreateTracker

def AnalizeOcupiedZone(trackers,flagTrackers,frame,tracker_type,DrawFrame):
    
    for Index in range(0, len(flagTrackers)):
        if(flagTrackers[Index]):
            ok, bbox = trackers[Index].update(frame)
            if ok:
                print("!!!!!!!!!!!El Sigue OCUPANDO la ZONA #: "+str(Index))
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                ########
                # VISUALICE ZONES ON IMAGE. CAN BE REMOVE
                cv2.rectangle(DrawFrame, p1, p2, (255, 0, 0), 2, 1)
                flagTrackers[Index] = True
                cv2.putText(DrawFrame, str(Index), (int(bbox[0]+bbox[2]/2), int(
                    bbox[1] + bbox[3]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 125, 155), 2)
                ########

            else:
                print("!!!!!!!!!!!El auto ha salido de la ZONA #: "+str(Index))
                flagTrackers[Index] = False
                trackers[Index] = CreateTracker(tracker_type)
 
 
def WatchIfCarOnZone(r,trackers,flagTrackers,frame,DrawFrame,MinDistance,Factor,Zone,pts):
    
    masks = r['masks']
    if(len(r['scores']) != 0):
            for scores, clases_id, detections, i in zip(r['scores'], r['class_ids'], r['rois'], range(0, len(masks[0, 0, :]))):

                pixelamount = 0
                pixelsonZone = np.zeros([Zone, 1])
                distanceToZone = np.zeros([4, Zone])
                Contador = 0
                Index = -1
                
                for r in range(0, Zone):
                    Contador = 0
                    pts3 = pts[r]
                    # Esquina superior izquierda
                    dx1 = detections[1] - pts3[0][0]
                    dy1 = detections[0]-pts3[0][1]
                    # Esquina inferior derecha
                    dx2 = detections[3] - pts3[1][0]
                    dy2 = detections[2]-pts3[1][1]
                    # Esquina superior derecha
                    dx3 = dx2
                    dy3 = dy1

                    # Esquina inferior izq
                    dx4 = dx1
                    dy4 = dy2

                    distanceToZone[0][r] = math.hypot(dy1, dx1)
                    if(distanceToZone[0][r] < MinDistance):
                        Contador = Contador+1

                    distanceToZone[1][r] = math.hypot(dy2, dx2)
                    if(distanceToZone[1][r] < MinDistance):
                        Contador = Contador+1

                    distanceToZone[2][r] = math.hypot(dy3, dx3)
                    if(distanceToZone[2][r] < MinDistance):
                        Contador = Contador+1

                    distanceToZone[3][r] = math.hypot(dy4, dx4)
                    if(distanceToZone[3][r] < MinDistance):
                        Contador = Contador+1
                    if(Contador == 4):
                        Index = r

                Flag = True
                if(Index != -1):
                    if(flagTrackers[Index] == False):
                        print(
                            "!!!!!!!!!!!El auto ha ENTRADO A la ZONA #: " + str(Index))

                        bbox = (int(detections[1]+(detections[3]-detections[1])*Factor), int(detections[0]+(detections[2]-detections[0])*Factor), int(
                            (detections[3]-detections[1])*(1-2*Factor)), int((detections[2]-detections[0])*(1-2*Factor)))

                        p1 = (int(bbox[0]), int(bbox[1]))

                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        cv2.rectangle(DrawFrame, p1, p2, (255, 0, 0), 2, 1)
                        trackers[Index].init(frame, bbox)
                        flagTrackers[Index] = True

