import pyeeg
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#import scikit-learn as sklearn

#fid =


dataPointsTest = [[]]
labels = []


#features
dfaOpen = []
hurstOpen = []
pfdOpen = []
stdOpen = []

dfaClosed = []
hurstClosed = []
pfdClosed = []
stdClosed = []
#specEntropyOpen = []


eyesOpen = [] #dataset A
eyesClosed = [] #dataset B

def main():

    partialPathZ = "c:\Users\Adrian Ribe\Desktop\Masteroppgave\Code\Machine learning trial\Z\Z"
    partialPathO = "c:\Users\Adrian Ribe\Desktop\Masteroppgave\Code\Machine learning trial\O\O"

#To read and extract features from eyes open
    for i in range(10):
        index = "{:03}".format(i + 91)
        filepath = partialPathZ + str(index) + ".txt"
        i+= 1
        file = open(filepath, 'r')
        tmp = file.readlines()
        dataPoint = [float(k) for k in tmp]
        dfaOpen.append(pyeeg.dfa(dataPoint))
        #hurstOpen.append(pyeeg.hurst(dataPoint))
        pfdOpen.append(pyeeg.pfd(dataPoint))
        stdOpen.append(np.std(dataPoint))

        #specEntropyOpen.append(pyeeg.spectral_entropy(dataPoint))

#To read and extract features from eyes closed
    for i in range(10):
        index = "{:03}".format(i + 91)
        filepath = partialPathO + str(index) + ".txt"
        i+= 1
        file = open(filepath, 'r')
        tmp = file.readlines()
        dataPoint = [float(k) for k in tmp]
        dfaClosed.append(pyeeg.dfa(dataPoint))
        #hurstClosed.append(pyeeg.hurst(dataPoint))
        pfdClosed.append(pyeeg.pfd(dataPoint))
        stdClosed.append(np.std(dataPoint))

        #specEntropyClosed.append(pyeeg.spectral_entropy(dataPoint))


    createDataset()
    #saveMachinestate(clf)
    predict()




def createDataset():
    global dfaOpen, dfaClosed, pfdOpen, pfdClosed, labels, dataPointsTest
    featureVector = []

    for i in range(len(dfaOpen)):
        featureVector = [dfaOpen[i], stdOpen[i]]
        dataPointsTest.append(featureVector)
        #labels.append(0)
        featureVector = [dfaClosed[i], stdClosed[i]]
        dataPointsTest.append(featureVector)
        #labels.append(1)

    dataPointsTest.pop(0) #remove empty list at start position

def predict():
    global dataPointsTest, labels

    Xtest = np.array(dataPointsTest)
    Xscaled = preprocessing.scale(Xtest)
    #min_max_scaler = preprocessing.MinMaxScaler()
    #X_test_minmax = min_max_scaler.fit_transform(Xtest)

    clf = joblib.load('testMachineState.pkl')#loads the machine-learning state
    

    print(clf.predict(Xscaled))


def saveMachinestate(clf):
    joblib.dump(clf, 'testMachineState.pkl')


if __name__ == '__main__':
	main()
