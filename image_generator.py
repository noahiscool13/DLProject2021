import math
from itertools import product
from random import choice, randint, choices

import numpy as np
from PIL import Image, ImageDraw


# w, h = 220, 190
# shape = [(40, 40), (w - 10, h - 10)]
#
# # creating new Image object
# img = Image.new("L", (w, h), (0))
#
# # create line image
# img1 = ImageDraw.Draw(img)
# img1.line(shape,fill=255, width=2)
# img.show()

def create_image():
    img = Image.new("L", (100, 100), (0))
    img1 = ImageDraw.Draw(img)
    corners = []
    for _ in range(choice([3])):
        start = (randint(0, 99), randint(0, 99))
        line_angle = randint(0, 359)
        length = choice([10, 20, 30])
        end = (int(start[0] + length * math.cos(line_angle)), int(start[1] + length * math.sin(line_angle)))
        img1.line((start, end), fill=255, width=1)
        for x in range(choice([2, 3, 4])):
            corners.append(end)
            start = end
            # line_angle = math.atan2(end[0]-start[0],end[1]-start[1])
            line_angle = line_angle + choice(
                [math.pi / 2, math.pi / 4, math.pi / 4 * 3, -math.pi / 2, -math.pi / 4, -math.pi / 4 * 3])
            length = choice(list(range(10, 40)))
            end = (int(start[0] + length * math.cos(line_angle)), int(start[1] + length * math.sin(line_angle)))
            img1.line((start, end), fill=255, width=1)
    return img, [corner for corner in corners if 0 <= corner[0] < 100 and 0 <= corner[1] < 100]


def slice(img, corners):
    good = []
    bad = []
    added_empty = 0
    for x in range(0,100-7,2):
        for y in range(0,100-7,2):
            crop = img.crop((x,y,x+8,y+8))
            if any((x+n[0],y+n[1]) in corners for n in product(list(range(2,6)),repeat=2)):
                good.append(crop)
            else:
                if np.array(crop).sum()>0 or 1:
                    bad.append(crop)
                else:
                    if added_empty<3:
                        bad.append(crop)
                        added_empty += 1

    if len(bad)>len(good):
        bad = choices(bad,k=len(good))
    elif len(bad)<len(good):
        good = choices(good,k=len(bad))

    # for x in range(10):
    #     bad[x].show()

    return [(x,1) for x in good]+[(x,0) for x in bad]

if __name__ == '__main__':

    print(slice(*create_image()))
