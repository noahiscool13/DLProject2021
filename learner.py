from random import shuffle

import numpy as np
import torch
from tqdm import tqdm

from image_generator import *


import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt


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

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.00005, momentum=0.9)


data_inp = slice(*create_image())
#8000 works
for x in tqdm(range(1_000)):
    data_inp.extend(slice(*create_image()))

print(len(data_inp))

data_test = slice(*create_image())
for x in range(100):
    data_test.extend(slice(*create_image()))

print(len(data_inp),len(data_test))

loss_lists = {"samples_seen":[],"train":[],"test":[],"epoch":[]}

for epoch in range(4):  # loop over the dataset multiple times
    shuffle(data_inp)
    running_loss = 0.0
    for i, data in enumerate(data_inp):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = torch.tensor((np.array(inputs).astype(np.float32)).reshape((64,)))
        labels = torch.tensor((float(labels),))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 4000 == 3999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 4000))
            loss_lists["train"].append(running_loss / 4000)
            loss_lists["samples_seen"].append(i+1+epoch*len(data_inp))
            running_loss = 0.0

            with torch.no_grad():
                tot_loss = 0
                for data in data_test:
                    inputs, labels = data
                    inputs = torch.tensor((np.array(inputs).astype(np.float32)).reshape((64,)))
                    labels = torch.tensor((float(labels),))
                    outputs = net(inputs)
                    tot_loss += float(criterion(outputs, labels))
                print(tot_loss/len(data_test))
                loss_lists["test"].append(tot_loss/len(data_test))
    loss_lists["epoch"].append((epoch+1)*len(data_inp))

print('Finished Training')

good = 0
bad = 0
cats = [0,0,0,0]
with torch.no_grad():
    for data in data_test:
        inputs, labels = data
        inputs = torch.tensor((np.array(inputs).astype(np.float32)).reshape((64,)))
        outputs = net(inputs)
        pred = int(outputs[0]>0.0)
        # print(pred,labels)
        if labels == pred:
            good+=1
        else:
            bad+=1

        cats[labels+pred*2] += 1

print(cats)
print(good,bad,good/(good+bad))

torch.save(net.state_dict(), "Trained_bad_1.pt")

plt.plot(loss_lists["samples_seen"],loss_lists["train"],label="Train")
plt.plot(loss_lists["samples_seen"],loss_lists["test"],label="Test")

y_min = min(min(loss_lists["train"]),min(loss_lists["test"]))
y_max = max(max(loss_lists["train"]),max(loss_lists["test"]))

plt.ylabel("Loss")
plt.xlabel("Samples seen")

for x in loss_lists["epoch"]:
    plt.plot([x,x],[y_min,y_max])
plt.legend()
# plt.show()
plt.savefig("learning_curve_3.png")