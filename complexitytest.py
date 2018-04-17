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


classifierstring = 'AllFeatures'

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def calculateFeature(i, InputData):
    #print(InputData)
    return features.FUNC_MAP[i](InputData)
def main():
    featuremask = features.readFeatureMask(classifierstring)
    nrOfFeatures = len(featuremask)
    low = -100
    high = 100
    dictionary = {}
    featureName = None
    InputData = [[]]
    timingResults = []

    X1, y1 = dataset.loadDataset(filename="data.txt", filterCondition=True,
                                filterType="DcNotch", removePadding=True, shift=False, windowLength=250)
    X1, y1 = dataset.sortDataset(X1, y1, length=130, classes=[0,1,2,3,4,5,6,7,8,9], merge = True) #,6,4,2,8
    #Just append some movement into the input data from each channel
    for i in range(8):
        InputData.append(X1[i][0])
    #Pop the first list because it is empty
    InputData.pop(0)

    #just iterate through all the features
    for i, v in enumerate(featuremask):
        wrapped = wrapper(calculateFeature, i, InputData)
        timingResults.append(min(timeit.repeat(wrapped, repeat = 50, number = 1)))
        featureString = str(features.FUNC_MAP.get(i))
        featureString = featureString.split(" ")
        featureName = featureString [1] + " "
        dictionary[featureName] = timingResults[i]
    sortedValues = sorted(list(dictionary.values()))
    sortedKeys = sorted(list(dictionary),  key = dictionary.__getitem__)

    print("\n")
    for i in range(len(sortedValues)):
        print(sortedKeys[i])
        print("%f \n" %sortedValues[i])



if __name__ == '__main__':
	main()
