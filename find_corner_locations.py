from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from image_generator import *
from PIL import Image, ImageDraw, ImageFilter





class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(64, 16)  # 5*5 from image dimension
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
net.load_state_dict(torch.load("Trained_v2.pt"))
net.eval()

def blur(a):
    kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum

def flood_fill_conected(arr):
    corners = []
    for x in range(arr.shape[1]):
        for y in range(arr.shape[0]):
            if arr[y][x]:
                locs = []
                to_check = [(x,y)]
                while to_check:
                    c = to_check.pop()
                    if arr[c[1]][c[0]]:
                        arr[c[1]][c[0]] = False
                        locs.append(c)
                        for off in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,-1)]:
                            if 0<=c[0]+off[0]<arr.shape[1] and 0<=c[1]+off[1]<arr.shape[0]:
                                to_check.append((c[0]+off[0],c[1]+off[1]))
                xs,ys = zip(*locs)
                if len(xs)>0:
                    corners.append((int(sum(xs)/len(xs)),int(sum(ys)/len(ys))))
    return corners


def find_corners(img):
    with torch.no_grad():
        outp = []

        for y in range(0, img.height - 7, 1):
            outp.append([])
            for x in range(0, img.width - 7, 1):
                crop = img.crop((x, y, x + 8, y + 8))
                inputs = torch.tensor((np.array(crop).astype(np.float32)).reshape((64,)))
                outputs = net(inputs)
                pred = int(outputs[0] > 1)
                outp[-1].append(pred)

        outp = blur(blur(outp))
        outp = outp>0.6
        corners = flood_fill_conected(deepcopy(outp))
        print(corners)
        print(len(corners))

        outp = Image.fromarray(np.array(outp)*255)

        imgDraw = outp.convert("RGB")
        img1 = ImageDraw.Draw(imgDraw)
        for corner in corners:
            img1.point(corner)
        outp.show()
        img.show()
        return corners


# find_corners(create_image()[0])