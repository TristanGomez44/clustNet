"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import cv2
import numpy as np
import sys
import torch
from torch.optim import SGD
from torchvision import models
from torchvision import datasets, transforms

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import netBuilder
from skimage.transform import resize
from matplotlib.mlab import PCA

#Look for the most important pixels in the activation value using mask
def salMap_mask(image,model,imgInd, maskSize=(1,1)):
    print("Computing saliency map using mask")
    pred,_ = model(image)

    argMaxpred = np.argmax(pred[0].detach().numpy())

    salMap = np.zeros((image.size()[2],image.size()[3]))

    for i in range(int(image.size()[2]/maskSize[0])):
        for j in range(int(image.size()[3]/maskSize[1])):

            maskedImage = torch.tensor(image)


            xPosS = i*maskSize[0]
            yPosS = j*maskSize[1]

            xPosE = min(image.size()[2],xPosS+maskSize[0])
            yPosE = min(image.size()[3],yPosS+maskSize[1])

            """
            print("------ New mask ------")
            print(maskedImage[0][0][xPosS:xPosE])
            print(yPosS)
            print(maskedImage[0][0][xPosS:xPosE,yPosS])
            """
            maskedImage[0][0][xPosS:xPosE,yPosS:yPosE] = image.min()
            maskedPred,_ = model(maskedImage)

            err = torch.pow((pred[0][argMaxpred] - maskedPred[0][argMaxpred]),2)
            salMap[xPosS:xPosE,yPosS:yPosE] = err.detach().numpy()

    writeImg("../vis/salMapMask_img_{}_u{}.png".format(imgInd,argMaxpred),salMap)

#Look for the most important pixels in the activation value using derivative
def salMap_der(image,model,imgInd):
    print("Computing saliency map using derivative")
    pred,_ = model(image)

    argMaxpred = np.argmax(pred.detach().numpy())

    loss = - pred[0][argMaxpred]

    loss.backward()

    salMap = image.grad/image.grad.sum()

    writeImg("../vis/salMapDer_img_{}_u{}.png".format(imgInd,argMaxpred),salMap.numpy()[0,0])

def opt(image,model,imgInd, layInd, unitInd, epoch=10000, nbPrint=4, alpha=6, beta=2,
        C=20, B=2, stopThre = 0.1):
    print("Maximizing activation")

    model.eval()
    Bp = 2*B
    V=B/2
    optimizer = SGD([image], lr=0.05)

    i=1
    lastVar = stopThre
    last_img = np.copy(image.detach().numpy())
    while i<epoch and lastVar >= stopThre:

    #for i in range(1, epoch+1):
        optimizer.zero_grad()

        _,activArr = model(image)

        # Loss function is minus the mean of the output of the selected layer/filter
        act = -torch.mean(activArr[layInd][0][unitInd])

        # computing the norm : +infinity if one pixel is above the limit,
        # else, computing a soft-constraint, the alpha norm (raised to the alpha power)
        if image.detach().numpy()[0,0].any() > Bp:
            norm = torch.tensor(float("inf"))
        else:
            norm = torch.sum(torch.pow(image,alpha))/float(image.size()[2]*image.size()[3]*np.power(B,alpha))

        # computing TV
        h_x = image.size()[2]
        w_x = image.size()[3]
        h_tv = torch.pow((image[:,:,1:,1:]-image[:,:,:h_x-1,:w_x-1]),2)
        w_tv = torch.pow((image[:,:,1:,1:]-image[:,:,:h_x-1,:w_x-1]),2)
        tv =  torch.pow(h_tv+w_tv,beta/2).sum()/(h_x*w_x*np.power(V,beta))

        loss = C*act+norm+tv

        # Backward
        loss.backward()
        # Update image
        optimizer.step()

        if i % int(epoch/(nbPrint-1)) == 0:
            np_img = np.copy(image.detach().numpy())
            lastVar = np.power(last_img - np_img,2).sum()/np.power(last_img,2).sum()
            last_img = np.copy(image.detach().numpy())
            # Save image
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            writeImg('../vis/img_'+str(imgInd)+'layer_vis_l' + str(layInd) +'_u' + str(unitInd) + '_iter'+str(i)+'.jpg',image.detach().numpy()[0,0])
        i += 1

def writeImg(path,img):

    np_img = img+np.abs(img.min())

    np_img = (255*np_img/np_img.max()).astype('int')
    #print(np_img)
    np_img = resize(np_img,(300,300),mode="constant", order=0)

    np_img = np_img+np.abs(np_img.min())
    np_img = (255*np_img/np_img.max()).astype('int')

    cv2.imwrite(path,np_img)

def getContr(task):
    if task == "ClustDetectNet":
        return netBuilder.ClustDetectNet
    elif task == "Net":
        return netBuilder.Net
    elif task == "OneClNet":
        return netBuilder.OneClNet
    else:
        print("Unknown net type : {}. Returning None".format(modelType))
        return None

def main():

    if len(sys.argv) != 6:
        print("Usage : vis.py <pathToModel> <modelType> <nbImages> <layInd> <unitInd>")
        sys.exit(0)

    modelPath = sys.argv[1]
    task = sys.argv[2]
    nbImages = int(sys.argv[3])
    layInd = int(sys.argv[4])
    unitInd = int(sys.argv[5])

    model = getContr(task)(inputSize=28,nbCl=5,inputChan=1)
    model.load_state_dict(torch.load(modelPath))

    print("ClustWeight")
    for p in model.getClustWeights():
        print(p)

    print("DetectWeight")
    for p in model.getDetectWeights():
        print(p)

    #Loading the test set
    kwargs = {}
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/MNIST', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=1, shuffle=True, **kwargs)

    #Comouting image that maximises activation of the given unit in the given layer
    maxInd = len(test_loader.dataset) - 1
    i = 0
    model.eval()
    while i < nbImages:
        print("Image ",i)

        img = Variable(test_loader.dataset[i][0]).unsqueeze(0)

        writeImg('../vis/img_'+str(i)+'.jpg',test_loader.dataset[i][0][0].detach().numpy())

        img.requires_grad = True

        opt(img,model,i,layInd,unitInd)
        #salMap_der(img,model,i)
        #salMap_mask(img,model,i)
        i +=1
