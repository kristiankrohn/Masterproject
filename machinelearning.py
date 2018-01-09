from sklearn import datasets, neighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.externals import joblib #in order to store the svm
import pandas as pd
import seaborn as sns
sns.set_palette('husl')

DataSet = []
predictSet = []
DataSetLabels = [] #r1 = 0, u1 = 1, l1 = 2, d1 = 3, c1 = 4
convertedDataSet = [[]]
convertedCh0 = [[]]
convertedCh1 = [[]]
convertedPredictData = [[]]



def main():
    AllData, predictData = readFile()
    arrangeDataset(AllData, predictData)
    extractFeatures() #calculate different features of time-series.
    #clf = createAndTrain() #switch to loadAndTrain() if you want to load previous machine-learning-state
    #saveMachinestate(clf) #if you want to save the state of classifier
    #predict(clf) #predict with given classifier.


def readFile():

    file = open('data.txt', 'r')
    pFile = open('data.txt', 'r')
    AllData = file.read()
    predictData = pFile.read()
    return AllData, predictData





def arrangeDataset(AllData, predictData):
    global Dataset, predictSet, convertedDataSet, convertedPredictData, convertedCh0, convertedCh1

    DataSet = AllData.split(':')
    predictSet = predictData.split(':')


    for i in range(len(DataSet)):
        feature = []
        feature = DataSet[i].split(',')

        featuretype = feature[0]
        feature.pop(0)
        convertedDataSet.append(map(float, feature))
        chIndex = getChannelIndexAndAppendLabel(featuretype)
        if chIndex == 0:
            convertedCh0.append(map(float, feature))
        if chIndex == 1:
            convertedCh1.append(map(float, feature))


        #pops the two empty lists at index 0 and end-index
        if i == (len(DataSet) - 1):
            convertedDataSet.pop(0)
            convertedCh0.pop(0)
            convertedCh1.pop(0)
            convertedDataSet.pop(len(DataSet) -1)

            #print(convertedDataSet)
    for i in range(len(predictSet)):
            predictionFeature = [] #this is just for trying to predict a given dataset
            predictionFeature = predictSet[i].split(',')
            predictionFeature.pop(0)

            convertedPredictData.append(map(float, predictionFeature))


            #pops the two empty lists at index 0 and end-index
            if i == (len(predictSet) - 1):
                convertedPredictData.pop(0)
                convertedPredictData.pop(len(predictSet) - 1)



def getChannelIndexAndAppendLabel(featuretype):
    global DataSetLabels

    if featuretype == 'r1' or featuretype == 'r0':
        DataSetLabels.append(0)
    if featuretype == 'u1' or featuretype == 'u0':
        DataSetLabels.append(1)
    if featuretype == 'l1' or featuretype == 'l0':
        DataSetLabels.append(2)
    if featuretype == 'd1' or featuretype == 'd0':
        DataSetLabels.append(3)
    if featuretype == 'c1' or featuretype == 'c0':
        DataSetLabels.append(4)

    if '0' in featuretype:
        return 0
    if '1' in featuretype:
        return 1


def createAndTrain():
    global convertedDataSet, convertedPredictData, DataSetLabels

    X = np.array(convertedDataSet)

    Y = np.array(map(float, DataSetLabels))
    #print(Y)
    clf = LinearSVC()
    clf.fit(X,Y)
    return clf


def loadAndTrain():
    global convertedDataSet, DataSetLabels

    X = np.array(convertedDataSet)
    Y = np.array(map(float, DataSetLabels))


    clf = joblib.load('machinestate.pkl')#loads the machine-learning state
    clf.fit(X,Y)
    return clf

def saveMachinestate(clf):
    joblib.dump(clf, 'machinestate.pkl')


def predict(clf):
    global convertedPredictData

    Z = np.array(convertedPredictData)
    print(len(Z))
    print(clf.predict(Z))#if this doesn't work, check readFile().


def extractFeatures():
    global convertedDataSet, convertedCh0, convertedCh1
    #print(convertedDataSet)
    std = np.std(convertedDataSet, axis = 1) #prints the standard-deviation of the 500 samples, for all movementdirections
    #cov = np.cov(convertedDataSet, rowvar = True)

    mean = np.mean(convertedDataSet, axis = 1) #mean when not looking at channels seperately
    meanCh0 = np.mean(convertedCh0, axis = 1)
    meanCh1 = np.mean(convertedCh1, axis = 1)
    var = np.var(convertedDataSet, axis = 1) #var when not looking at channels seperately
    varCh0 = np.var(convertedCh0, axis = 1)
    varCh1 = np.var(convertedCh0, axis = 1)
    minimum = np.amin(convertedDataSet, axis = 1) #minimum when not looking at channels seperately
    minimumCh0 = np.amin(convertedCh0, axis = 1)
    minimumCh1 = np.amin(convertedCh1, axis = 1)
    maximum = np.amax(convertedDataSet, axis = 1)#maximum when not looking at channels seperately
    maximumCh0 = np.amax(convertedCh0, axis = 1)
    maximumCh1 = np.amax(convertedCh1, axis = 1)

    #absolute = np.absolute(convertedDataSet) Blir et 2D-array fordi den tar absoluttverdien til alle samples, for alle retninger. What to do?
    compMean = np.subtract(np.mean(convertedCh0, axis = 1), np.mean(convertedCh1, axis = 1))
    compStd = np.subtract(np.std(convertedCh0, axis = 1), np.std(convertedCh1, axis = 1))
    compVar = np.subtract(np.var(convertedCh0, axis = 1), np.var(convertedCh1, axis = 1))
    compMinimum = np.subtract(np.amin(convertedCh0, axis = 1), np.amin(convertedCh1, axis = 1))
    compMaximum = np.subtract(np.amax(convertedCh0, axis = 1), np.amax(convertedCh1, axis = 1))
    #compIndexMinimum = np.fmin(convertedCh0, convertedCh1)
    #corr = np.correlate(convertedCh0, convertedCh1)
    


    #predictFeatures = [mean, var, min, max, compMean, compStd, compVar, compMin, compMax]

    #from here and down is for visualization purposes only.

    visualizeFeature(std, "standard-deviation") #need to have "compare" in string if you want to compare channels

def visualizeFeature(visu, string):
    global DataSetLabels

    visuValue = []
    visuDirection = []




    Y = np.array(map(float, DataSetLabels))

    if 'compare' in string:
        for i in range(len(Y)/2):
            visuValue.append(visu[i])
            if Y[i] == 0:
                visuDirection.append('Right')

            if Y[i] == 1:
                visuDirection.append('Up')

            if Y[i] == 2:
                visuDirection.append('Left')

            if Y[i] == 3:
                visuDirection.append('Down')

            if Y[i] == 4:
                visuDirection.append('Center')

    else:
        for i in range(len(Y)):
            visuValue.append(visu[i])
            if Y[i] == 0:
                visuDirection.append('Right')

            if Y[i] == 1:
                visuDirection.append('Up')

            if Y[i] == 2:
                visuDirection.append('Left')

            if Y[i] == 3:
                visuDirection.append('Down')

            if Y[i] == 4:
                visuDirection.append('Center')




    dictionary = {'Direction': visuDirection, '%s' % string: visuValue}
    dataframe = pd.DataFrame(dictionary, columns = ['Direction', '%s' % string])
    #print(dataframe)
    g = sns.violinplot(y = dataframe.Direction, x = '%s' % string, data=dataframe, inner='point')
    savestring = "featureFigures/" + string + ".png"
    plt.savefig(savestring, bbox_inches='tight')
    plt.close()

    #numberOfMin = len(np.argmin(convertedDataSet, axis = 1))
    #numberOfMax = len(np.argmax(convertedDataSet, axis = 1))







#print(convertedPredictData)
#X = np.array(convertedDataSet)

#Y = np.array(map(float, DataSetLabels))

#Z = np.array(convertedPredictData)


#clf = LinearSVC()



#joblib.dump(clf, 'machineStateDataSet.pkl')#saves the machine-learning state



if __name__ == '__main__':
	main()
