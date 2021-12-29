#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:58:15 2021

@author: nina.hosseinikivanani
"""

#%%
import torch
from torch import optim,nn
import torch.nn.functional as F
from torchvision import transforms as T,models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.hub import load_state_dict_from_url

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from operator import itemgetter
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import optim,nn
import torch.nn.functional as F
from torchvision import transforms as T,models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid

from sklearn.metrics import precision_score, recall_score, f1_score
#pd.options.plotting.backend = "plotly"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
data = pd.read_csv('/dataset/onehot_combined_k5.csv')

#print(data)


#%%
class NA_Dataset(Dataset):

    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir 
        self.transform = transform 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file = self.img_dir + self.data.iloc[:,0][idx]
        #print("Print it here:", img_file)
        img = Image.open(img_file).convert('RGB')
        img = img.resize((224,224))
        label = np.array(self.data.iloc[:,1:].iloc[idx])

        if self.transform:
            img = self.transform(img)

        return img,label


#%%
trainds = NA_Dataset(data,
                      img_dir = '/dataset/sample/K5_DataSet_SMOTEonly/',
                      transform = T.Compose([T.ToTensor()]))

#print(next(iter(trainds)))



#%%Check inconsistency
# img_file = []
# path = '/dataset/sample/K5_DataSet_SMOTEonly'



# files = os.listdir(path)

# for f in files:
#   img_file.append(f.split(".png")[0])
# print(img_file)
# len(img_file)

#%%
trainset, validset, testset = random_split(trainds, [int(len(data)*0.7),int(len(data)*0.1),(len(data)-int(len(data)*0.7)-int(len(data)*0.1))]) #the number of images needed to be changed

print("Length of trainset : {}".format(len(trainset)))
print("Length of testset : {}".format(len(testset)))
print("Length of validset : {}".format(len(validset)))


#%%
trainloader = DataLoader(trainset,
                         batch_size = 32,
                         shuffle = True)

validloader = DataLoader(validset,
                         batch_size = 32,
                         shuffle = False)

testloader = DataLoader(testset,
                        batch_size = 32,
                        shuffle = True)


#%%
#import torch
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

# Check the output without pretrained model for resnet34 and resnet50
#2 model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
#3 model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)

# or any of these variants

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()
#print(model)

#%%
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(512, 15),
    nn.Sigmoid()
)

model.to(device)

#%%
optimizer = optim.Adam(model.parameters(),
                       lr = 0.0001)
schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor = 0.1,
                                                 patience = 4)
epochs = 20
valid_loss_min = np.Inf
#%%
#criteria = nn.CrossEntropyLoss()

# def cross_entropy_one_hot(ps, labels):
#     print(labels)
#     _, labels = labels.max(dim = 0 )
#     #print(labels.shape)
#     return nn.CrossEntropyLoss()(ps, labels)

#criteria = nn.BCEWithLogitsLoss()
criteria = nn.BCELoss()

#%%
# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            }

#%%
for i in range(epochs):

    train_loss = 0.0
    valid_loss = 0.0
    train_acc = 0.0
    valid_acc = 0.0 

    model.train()
    for images,labels in tqdm(trainloader):
        images = images.to(device)
        #print("Images size", images.shape)
        labels = labels.to(device)
        labels = labels.float()
        #print(labels)
        #print("This is labels type", labels.dtype)
        #print("Labels size", labels.shape)
    

        ps = model(images)
        #print("this is ps type", ps.dtype)
        #print(ps.shape)
        #ps = ps.unsqueeze(0)
        loss = criteria(ps, labels.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    avg_train_loss = train_loss / len(trainloader)

    model.eval()
    with torch.no_grad():
        model_result =[]
        targets = []
        for images,labels in tqdm(validloader):
            images = images.to(device)
            labels_valid = labels.to(device)
            labels_valid = labels.float()
            ps_valid = model(images)
            loss = criteria(ps_valid, labels_valid.squeeze())
            #loss = weighted_loss(pos_weights,neg_weights,ps,labels)
            model_result.extend(ps_valid.cpu().numpy())
            targets.extend(labels_valid.cpu().numpy())
            valid_loss += loss.item()
        avg_valid_loss = valid_loss / len(validloader)

    schedular.step(avg_valid_loss)

    result = calculate_metrics(np.array(model_result), np.array(targets))
    print(result)
    if avg_valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).   Saving model ...'.format(valid_loss_min,avg_valid_loss))
        torch.save({
            'epoch' : i,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'valid_loss_min' : avg_valid_loss
        },'Image_model.txt')

        valid_loss_min = avg_valid_loss

    print("Epoch : {} Train Loss : {:.6f} ".format(i+1,avg_train_loss))
    print("Epoch : {} Valid Loss : {:.6f} ".format(i+1,avg_valid_loss))
    


#%%
