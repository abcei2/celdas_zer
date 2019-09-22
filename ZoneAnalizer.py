import cv2
import math
import numpy as np


def WatchIfCarOnZone(r, frame, DrawFrame, MinDistance, Factor, Zone, pts):
    if len(r['scores']) == 0:
        return

    masks = r['masks']
    zip_thing = zip(
        r['scores'],
        r['class_ids'],
        r['rois'],
        range(0, len(masks[0, 0, :]))
    )
    for scores, clases_id, match, i in zip_thing:
        distanceToZone = np.zeros([4, Zone])
        Contador = 0
        Index = -1

        for r in range(0, Zone):
            Contador = 0
            pts3 = pts[r]
            # Esquina superior izquierda
            dx1 = match[1] - pts3[0][0]
            dy1 = match[0] - pts3[0][1]
            # Esquina inferior derecha
            dx2 = match[3] - pts3[1][0]
            dy2 = match[2] - pts3[1][1]

            # Esquina superior derecha
            dx3 = dx2
            dy3 = dy1
            # Esquina inferior izq
            dx4 = dx1
            dy4 = dy2

            distanceToZone[0][r] = math.hypot(dy1, dx1)
            if(distanceToZone[0][r] < MinDistance):
                Contador += 1

            distanceToZone[1][r] = math.hypot(dy2, dx2)
            if(distanceToZone[1][r] < MinDistance):
                Contador += 1

            distanceToZone[2][r] = math.hypot(dy3, dx3)
            if(distanceToZone[2][r] < MinDistance):
                Contador += 1

            distanceToZone[3][r] = math.hypot(dy4, dx4)
            if(distanceToZone[3][r] < MinDistance):
                Contador += 1

            if(Contador == 4):
                Index = r

            print("distanceToZone", distanceToZone)

        if Index != -1:
            print(f"!!!!!!!!!El auto ha [[ENTRADO]] A la ZONA #: {Index}")
            bbox = (
                int(match[1] + (match[3] - match[1]) * Factor),
                int(match[0] + (match[2] - match[0]) * Factor),
                int((match[3] - match[1]) * (1 - 2 * Factor)),
                int((match[2] - match[0]) * (1 - 2 * Factor))
            )
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(DrawFrame, p1, p2, (255, 0, 0), 2, 1)
        else:
            print(f"!!!!!!!!!El auto ha [[SALIDO]] de la ZONA #: {Index}")
