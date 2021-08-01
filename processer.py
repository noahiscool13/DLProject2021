from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from image_generator import *
from PIL import Image, ImageDraw



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
net.load_state_dict(torch.load("Trained_v1.pt"))
net.eval()


with torch.no_grad():
    img = create_image()[0]
    imgDraw = img.convert("RGB")
    img1 = ImageDraw.Draw(imgDraw)


    outp = []

    for y in range(0, 100 - 7, 1):
        outp.append([])
        for x in range(0, 100 - 7, 1):
            crop = img.crop((x, y, x + 8, y + 8))
            inputs = torch.tensor((np.array(crop).astype(np.float32)).reshape((64,)))
            outputs = net(inputs)
            pred = int(outputs[0] > 0.5)
            outp[-1].append(pred)
            if pred:
                img1.point((x+4,y+4),(255,0,0))

    imgDraw.show()
    outp = Image.fromarray(np.array(outp)*255)
    outp.show()

# 90%