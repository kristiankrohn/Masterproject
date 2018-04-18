import time
import features
import dataset
import numpy as np
import random
import pyeeg
from globalconst import  *
import globalvar as glb
from numpy.fft import fft
from numpy import zeros, floor
import math
import timeit



def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def calculateFeature(i, InputData):
    return features.FUNC_MAP[i](InputData)

def writeToFile(values, windowLength):
    file = open("executionTime" + slash + 'executionTimeAllFeatures' + str(windowLength) + '.txt', 'w')
    file.write(str(values))
    file.close()

def getExecutionCost(featuremask, executionTimeList):
    cost = 0
    for i in range(len(featuremask)):
        cost += executionTimeList[featuremask[i]]
    return cost

def readExecutionTime(filename = 'executionTimeAllFeatures250.txt'):
    file = open("executionTime" + slash + filename, 'r')
    executionTimeString = file.read()
    file.close()
    executionTimeString = executionTimeString[1:-1]
    executionTimeList = map(float, executionTimeString.split(', '))
    return executionTimeList

def printSortedValues(sortedKeys, sortedValues):
    print("\n")
    for i in range(len(sortedValues)):
        print(sortedKeys[i])
        print("%f \n" %sortedValues[i])

def calculateAndWriteExecutionTime(classifierstring = 'AllFeatures', shift = False, windowLength = 250, Sort = False):
    featuremask = features.readFeatureMask(classifierstring)
    nrOfFeatures = len(featuremask)
    dictionary = {}
    featureName = None
    InputData = [[]]
    executionTimeList = []

    X1, y1 = dataset.loadDataset(filename="data.txt", filterCondition=True,
                                filterType="DcNotch", removePadding=True, shift=shift, windowLength=windowLength)
    X1, y1 = dataset.sortDataset(X1, y1, length=130, classes=[0,1,2,3,4,5,6,7,8,9], merge = True) #,6,4,2,8
    #Just append some movement into the input data from each channel
    for i in range(8):
        InputData.append(X1[i][0])
    #Pop the first list because it is empty
    InputData.pop(0)
    #just iterate through all the features
    for i in range(len(featuremask)):
        wrapped = wrapper(calculateFeature, i, InputData)
        featureString = str(features.FUNC_MAP.get(i))
        featureString = featureString.split(" ")
        featureName = featureString [1] + " "
        dictionary[featureName] = min(timeit.repeat(wrapped, repeat = 10000, number = 1))
        print("finished with feature %d" %i)
        executionTimeList.append(dictionary[featureName])

    #writeToFile(executionTimeList, windowLength)
    #Should the list be sorted and printed?
    if Sort:
        sortedValues = sorted(list(dictionary.values()))
        sortedKeys = sorted(list(dictionary),  key = dictionary.__getitem__)
        printSortedValues(sortedKeys, sortedValues)




if __name__ == '__main__':
	#calculateAndWriteExecutionTime(windowLength = 100)
    print(readExecutionTime(filename = 'executionTimeAllFeatures100.txt'))
