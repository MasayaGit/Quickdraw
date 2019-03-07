
# coding: utf-8

# In[1]:


import numpy as np
import json


def appendData(argumentFileName,makedataXList,makedataYList,number,countNumber):
    file = '/home/QuickDraw/NumpyBitdata/' + argumentFileName
    numpyDataList = np.load(file)
    count = 0
    for numpyData in numpyDataList:
        #255で割って範囲を０−１にする
        if count == countNumber:
            break
        makedataXList.append(numpyData/255)
        makedataYList.append(number)
        count += 1
    return 


def makeDataList():
    makedataXList = []
    makedataYList = []
    appendData('airplane.npy',makedataXList,makedataYList,0,60000)
    appendData('apple.npy',makedataXList,makedataYList,1,60000)
    appendData('cat.npy',makedataXList,makedataYList,2,60000)
    appendData('flower.npy',makedataXList,makedataYList,3,60000)
    appendData('bus.npy',makedataXList,makedataYList,4,60000)
    return  makedataXList,makedataYList

