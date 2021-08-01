from math import sqrt, dist

import numpy as np
import cv2
from PIL import Image

from find_corner_locations import find_corners

skip = 100
cap = cv2.VideoCapture('VID_hi_slow.mp4')
corners = []

n = 0

result = cv2.VideoWriter('test.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (1000,500))

while (cap.isOpened()):
    n+=1
    print(n)
    ret, frame = cap.read()
    if not ret:
        break
    if skip:
        skip -= 1
        continue
    # n+=1
    # if n==10:
    #     break
    # if n%2==0:
    #     continue
    # frame = cv2.resize(frame, (800, frame.shape[0] * 800 // frame.shape[1]))
    frame = cv2.resize(frame, (1000, 500))


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    # edges = cv2.arrowedLine(edges, (20,20), (100,150),
    # 					  255, 4)
    new_corners = find_corners(Image.fromarray(edges))

    if new_corners and corners:
        for corner in new_corners:
            closest = min(corners, key=lambda x: dist(corner, x))
            if dist(closest,corner)<15:
                frame = cv2.arrowedLine(frame, corner, (
                corner[0] + (corner[0]-closest[0]) * 5, corner[1] + (corner[1]-closest[1]) * 5), (0, 255, 0), 2)
    corners = new_corners

    cv2.imshow('frame', frame)
    cv2.imshow('edges',edges)
    result.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
result.release()
cv2.destroyAllWindows()
