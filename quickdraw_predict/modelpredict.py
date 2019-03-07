import numpy as np
import cv2

import matplotlib 
matplotlib.use('tkagg') 

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

def judgeAndPlt(imageCV,resnet18,resnet34):
    img = cv2.resize(imageCV, dsize=(28, 28))
    img = cv2.bitwise_not(img)
    img_tensor = torch.from_numpy(img/255)
    #plt.imshow(img,cmap="gray")
    #plt.show()

    img_tensor = img_tensor.view(-1,1, 28, 28)
    img_tensor = img_tensor.float()
    img_tensor = img_tensor.to('cpu')
   
    # 評価モードにする 各resnetを書く
    resnet18 = resnet18.eval()
    resnet34 = resnet34.eval()

    softmax = []
    with torch.no_grad():
        output1 = resnet18(img_tensor)
        output2 = resnet34(img_tensor)
        alloutput = output1 + output2

        #print(output1)
        #print(output2)
        #print(alloutput)
        softmax = F.softmax(alloutput)

    softmax = softmax.numpy()
    #print(softmax)
    softmax = softmax.reshape(-1)

    predict = [int(l.argmax()) for l in alloutput]
    strpredict = "predict: " + softmaxResultToString(predict)
    label = ["airplane","apple","cat", "flower", "bus"]
    plt.pie(softmax,labels=label)
    plt.title(strpredict)
    #plt.text(1, 1,softmaxResultToString(predict))
    plt.show()

def softmaxResultToString(netPred):
    if netPred == [0]:
        return "airplane"
    if netPred == [1]:
        return "apple"
    if netPred == [2]:
        return "cat"
    if netPred == [3]:
        return "flower"
    if netPred == [4]:
        return "bus"