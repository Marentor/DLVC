import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reference.assignments.assignment1.dlvc import ops
from reference.assignments.assignment1.dlvc.batches import BatchGenerator
from reference.assignments.assignment1.dlvc.datasets.pets import PetsDataset
from reference.assignments.assignment1.dlvc.models.pytorch import CnnClassifier


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

op = ops.chain([
    ops.hwc2chw(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])
size_of_batch=128

p=PetsDataset(r"C:\Users\marti\PycharmProjects\pythonProject\cifar-10-batches-py",1)
training_Batches=BatchGenerator(p,size_of_batch,False,op)
p=PetsDataset(r"C:\Users\marti\PycharmProjects\pythonProject\cifar-10-batches-py",2)
validation_Batches=BatchGenerator(p,size_of_batch,False,op)
p=PetsDataset(r"C:\Users\marti\PycharmProjects\pythonProject\cifar-10-batches-py",3)
test_Batches=BatchGenerator(p,size_of_batch,False,op)



use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
net.to(device)

clf = CnnClassifier(net,(0,32,32,3),2,0.01,0.0001)

losses=[]
for epoch in range(100):
    print(epoch+1)
    running_loss = 0.
    last_loss = 0.
    for batch in training_Batches:
        data = batch.data
        labels = batch.label
        loss=(clf.train(data,labels))
        loss=loss.tolist()
        losses.append(loss)
        converted_losses=np.array(losses)
    print("train_loss", np.mean(converted_losses), "±", np.std(converted_losses))

    for v_batch in validation_Batches:
        data = batch.data
        labels = batch.label
        predictions=clf.predict(data)





