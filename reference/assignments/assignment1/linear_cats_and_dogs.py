from datetime import datetime

from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
import dlvc.ops as ops
import numpy as np
import torch


class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim=3072, num_classes=2):
    super(LinearClassifier, self).__init__()

    self.linear = torch.nn.Linear(input_dim, num_classes)

  def forward(self, x):
    x = self.linear(x)
    return x


op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])
size_of_batch=500
p=PetsDataset(r"C:\Users\marti\PycharmProjects\pythonProject\cifar-10-batches-py",1)
training_Batches=BatchGenerator(p,size_of_batch,False,op)
p=PetsDataset(r"C:\Users\marti\PycharmProjects\pythonProject\cifar-10-batches-py",2)
validation_Batches=BatchGenerator(p,size_of_batch,False,op)
p=PetsDataset(r"C:\Users\marti\PycharmProjects\pythonProject\cifar-10-batches-py",3)
test_Batches=BatchGenerator(p,size_of_batch,False,op)




model = LinearClassifier()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


'''
TODO: Train a model for multiple epochs, measure the classification accuracy on the validation dataset throughout the training and save the best performing model. 
After training, measure the classification accuracy of the best perfroming model on the test dataset. Document your findings in the report.


'''

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

best_Accuracy=0
n_epochs=100
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
for it in range(n_epochs):
    running_loss = 0.
    last_loss = 0.
    for batch in training_Batches:
        local_data=batch.data
        local_labels=batch.label
        local_data=torch.from_numpy(local_data)
        local_labels=torch.from_numpy(local_labels)
        # Transfer to GPU
        # local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        optimizer.zero_grad()
        outputs = model(local_data.float())
        # Compute the loss and its gradients
        loss = criterion(outputs, local_labels.long())
        loss.backward()
        # Adjust learning weights
        optimizer.step()


    model.train(False)
    correct = 0
    size_of_validation_set=0
    for v_batch in validation_Batches:
        v_data=v_batch.data
        v_labels=v_batch.label
        v_data=torch.from_numpy(v_data)
        v_labels=torch.from_numpy(v_labels)
        voutputs = model(v_data.float())
        vloss = criterion(voutputs, v_labels.long())
        _, predicted = torch.max(voutputs.data, 1)
        size_of_validation_set+=len(v_labels)
        correct += (predicted == v_labels).sum()

    accuracy = 100 * correct / size_of_validation_set
    # Track best performance, and save the model's state
    if accuracy > best_Accuracy:
        best_Accuracy = accuracy
        model_path = 'model_{}_{}'.format(timestamp, accuracy)
        torch.save(model, model_path)
        best_validation_accuracy=accuracy;

    print(f'In this epoch {it + 1}/{n_epochs}, Training loss: {loss.item():.4f}, Validation accuracy: {accuracy:.4f}')
    with open('results.csv', 'a+') as f:

        f.write(f' {it + 1},{loss.item():.4f},{accuracy:.4f}\n')
    model.train(True)


model = LinearClassifier()
model = torch.load(model_path)
model.train(False)
correct = 0
size_of_validation_set=0
for v_batch in test_Batches:
    v_data=v_batch.data
    v_labels=v_batch.label
    v_data=torch.from_numpy(v_data)
    v_labels=torch.from_numpy(v_labels)
    voutputs = model(v_data.float())
    vloss = criterion(voutputs, v_labels.long())
    _, predicted = torch.max(voutputs.data, 1)
    size_of_validation_set+=len(v_labels)
    correct += (predicted == v_labels).sum()

test_accuracy = 100 * correct / size_of_validation_set
print(f'Best Model  Validation Accuracy: {best_validation_accuracy:.4f} Test accuracy: {test_accuracy:.4f} ')
