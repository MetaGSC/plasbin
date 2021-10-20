import os
import sys
import re
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from Bio import SeqIO
from datetime import datetime
from numpy.random import randint
import torch
import torch.utils.data
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import matplotlib
import matplotlib.pyplot as plt


from Scripts.TrainingComponents.HDF5dataset import HDF5Dataset
from Scripts.TrainingComponents.WeightsCreator import make_weights_for_balanced_classes


# Add Cuda availability

# datapath = '/home/chamikanandasiri/Datasets/plasbin_2M'
# datapath = '/home/chamikanandasiri/Datasets/plasbin_100K'
datapath = '/home/chamikanandasiri/Datasets/plasbin_4K'



print(f'torch vesrion:{torch.__version__}  cuda availability:{torch.cuda.is_available()} with {torch.cuda.device_count()} GPU devices')

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader(DataLoader):
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device, batchSize, *args, **kwargs):
        super(DeviceDataLoader, self).__init__(dl, batchSize, *args, **kwargs)
        self.dl = dl
        self.device = device
        # for b in self.dl: 
        #   yield to_device(b, self.device)
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        # super(DeviceDataLoader, self).__iter__()
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
print(f'{device} set as the default device')

k = 7

inputFeatures = int((4**k) / 2) + 15
# inputFeatures = 2
layer_array = [512, 512, 256, 256]
outputSize = 2
momentum = 0.4
dropoutProb = 0.5
batchSize = 200
num_epochs = 20
opt_func = torch.optim.Adam
lr = 0.001
num_workers = 2

print('Importing the dataset....')
# trainingDataset = HDF5Dataset('/home/chamikanandasiri/Datasets/DNAML_Plasmid/DNAML_plasmid_train.h5', True)
# trainingDataset = HDF5Dataset(
#     '/home/chamikanandasiri/Test/Plasmid_0.1_Dataset.h5', True)
trainingDataset = HDF5Dataset(
    datapath, True, data_cache_size=100, label_threshold=13)
datasetsize = len(trainingDataset)

train_size = int(0.8 * len(trainingDataset))
val_size = len(trainingDataset) - train_size

print('\nSplitting Training/Validation datasets....')
train_ds, val_ds = random_split(trainingDataset, [train_size, val_size])
print('==== Dataset processed ====')


# weights = make_weights_for_balanced_classes(outputSize, tens)
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, train_size)

# train_dl = DataLoader(train_ds, batchSize, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=sampler)
# val_dl = DataLoader(val_ds, batchSize, num_workers=num_workers, pin_memory=True)

#=================================================================================================
train_dl = DataLoader(train_ds, batchSize, shuffle=True, num_workers=num_workers, pin_memory=True)
val_dl = DataLoader(val_ds, batchSize, num_workers=num_workers, pin_memory=True)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig('Figures/accuracies.png')
    plt.show()
    plt.clf()

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig('Figures/losses.png')
    plt.show()
    plt.clf()

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

class Model(nn.Module):
    def __init__(self, in_size, layer_array = [512, 512, 256, 256], out_size = 28):
        super().__init__()
        self.network = nn.Sequential(
          nn.Linear(in_size, layer_array[0]),
          nn.ReLU(),
          # nn.Dropout(dropoutProb),
          nn.Linear(layer_array[0], layer_array[1]),
          nn.ReLU(),
          # nn.Dropout(dropoutProb),
          nn.Linear(layer_array[1], layer_array[2]),
          nn.ReLU(),
          # nn.Dropout(dropoutProb),
          nn.Linear(layer_array[2], layer_array[3]),
          nn.ReLU(),
          # nn.Dropout(dropoutProb),
          nn.Linear(layer_array[3], out_size)
        )
        
    def forward(self, xb):
        softmax = nn.LogSoftmax(dim=0)
        return softmax(self.network(xb))

    def training_step(self, batch):
        values, labels, f_id = batch 
        values = values.to(device)
        labels = labels.to(device)
        out = self(values)                  # Generate predictions
        loss = nn.functional.nll_loss(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        values, labels, f_id = batch 
        values = values.to(device)
        labels = labels.to(device)
        out = self(values)                    # Generate predictions
        loss = nn.functional.nll_loss(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

# def predict(value, model):
#   xb = value
#   # //:TODO Check torch.stack
#   # xb = torch.stack(value)
#   yb = model(xb)
#   _, preds  = torch.max(yb, dim=0)
#   return preds

def predict(value, model):
    # Convert to a batch of 1
    xb = to_device(value, device)
    # Get predictions from model
    yb = model(xb)
    # print(yb)
    # Pick index with highest probability   
    _, preds  = torch.max(yb, dim=0)
    # print(torch.max(yb, dim=0)[1])
    # Retrieve the class label
    return preds.item()

def calculate_accuracy(testingDataset, testDatasetsize):
    cor = []
    incor = []
    for i in range(testDatasetsize):
        prediction = predict(testingDataset[i][0], model)
        label = testingDataset[i][1].item()
        # print(label)
        if(prediction == label):
            cor.append([label, prediction])
        else:
            incor.append([label, prediction])

    correct_df = pd.DataFrame(cor, columns=['label', 'prediction'])
    incorrect_df = pd.DataFrame(incor, columns=['label', 'prediction'])
    
    print("Correct test results", len(correct_df))
    print("Incorrect test results", len(incorrect_df))

    correct_df.to_csv("correct_df_results.csv")
    incorrect_df.to_csv("incorrect_df_results.csv")



model = Model(inputFeatures, layer_array, outputSize)
model = to_device(model, device)
model

model.double()
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

plot_accuracies(history)

plot_losses(history)


testingDataset = HDF5Dataset(
    datapath, False, data_cache_size=100, label_threshold=13)

testDatasetsize = len(testingDataset)

print("The Length of the test dataset is:- ," , testingDataset)

test_dl = DataLoader(testingDataset, batchSize, num_workers=4, pin_memory=True)

calculate_accuracy(testingDataset, testDatasetsize)


