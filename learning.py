import pyeeg
import numpy as np
from sklearn import svm
from sklearn import preprocessing
import itertools
import plot
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV


from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA

#from numpy.fft import fft
#from numpy import zeros, floor
import math
import time


import dataset
from globalconst import  *
import globalvar
'''
import copy
import mail
from itertools import permutations
from itertools import combinations
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from datetime import datetime
'''
import features
import predict
import classifier





def main():
    #partialPathZ = "c:\Users\Adrian Ribe\Desktop\Masteroppgave\Code\Machine learning trial\Z\Z"
    #partialPathO = "c:\Users\Adrian Ribe\Desktop\Masteroppgave\Code\Machine learning trial\O\O"
    #extractFeatures(partialPathZ, partialPathO)
    #features.cleanLogs()
    #features.compareFeatures(n_jobs=-1)
    #features.readLogs()
    #features.compareFeatures2(8)
    startLearning()


def startLearning():
    bestParams = []
    accuracyScore = []
    f1Score = []
    precision = []
    classificationReport = []

    #classifierstring = "learning260RBFsvm22Features"
    classifierstring = 'AllFeatures'

    #X, y = dataset.loadDataset("longdata.txt")
    '''
    X, y = dataset.loadDataset(filename="data.txt", filterCondition=True,
                                filterType="DcNotch", removePadding=True, shift=False, windowLength=250)
    '''

    dataset.setDatasetFolder(1)

    X1, y1 = dataset.loadDataset(filename="data.txt", filterCondition=True,
                                filterType="DcNotch", removePadding=True, shift=False, windowLength=250)
    X1, y1 = dataset.sortDataset(X1, y1, length=130, classes=[0,1,2,3,4,5,6,7,8,9], merge = True) #,6,4,2,8

    '''
    X1T, y1T = dataset.loadDataset(filename="data.txt", filterCondition=True,
                                filterType="DcNotch", removePadding=True, shift=False, windowLength=250)
    X1T, y1T = dataset.sortDataset(X1T, y1T, length=130, classes=[0,1,2,3,4,5,6,7,8,9], merge = True) #,6,4,2,8
    '''
    dataset.setDatasetFolder(2)

    X2, y2 = dataset.loadDataset(filename="data.txt", filterCondition=True,
                                filterType="DcNotch", removePadding=True, shift=True, windowLength=150)
    X2, y2 = dataset.sortDataset(X2, y2, length=130, classes=[0,1,2,3,4,5,6,7,8,9], merge = True, zeroClassMultiplier=1) #,6,4,2,8

    '''
    X2T, y2T = dataset.loadDataset(filename="data.txt", filterCondition=True,
                                filterType="DcNotch", removePadding=True, shift=False, windowLength=250)
    X2T, y2T = dataset.sortDataset(X2T, y2T, length=130, classes=[0,1,2,3,4,5,6,7,8,9], merge = True) #,6,4,2,8
    '''

    #X, y = dataset.sortDataset(X, y, length=10000, classes=[6,8], merge = False)


    #def sortDataset(x=None, y=None, length=10, classes=[0,5,4,2,6,8])
    #if x or y is undefined, data.txt will be loaded


    channelIndex = 0
    '''
    FUNC_MAP = {0: hfd,
            1: minDiff,
            2: maxDiff,
            3: specEntropy,
            4: pearsonCoeff14,
            5: stdDeviation,
            6: slope,
            7: thetaBeta1,
            8: extrema
            9: pearsonCoeff13}
    '''
    #XL = features.extractFeatures(X, channelIndex)
    featuremask = features.readFeatureMask(classifierstring)
    XL1 = features.extractFeaturesWithMask(
            X1, featuremask=featuremask, printTime=False)
    XL2 = features.extractFeaturesWithMask(
            X2, featuremask=featuremask, printTime=False)

    #If a test set is needed for the combined subject model
    '''
    XL1T = features.extractFeaturesWithMask(
            X1T, featuremask=featuremask, printTime=False)
    XL2T = features.extractFeaturesWithMask(
            X2T, featuremask=featuremask, printTime=False)
    '''
    #uncomment for using samples as features
    '''
    XL2 = X2[0]
    print(len(X2[0]))
    for i in range(len(X2[0])):
        XL2[i] = np.concatenate((XL2[i], X2[1][i], X2[3][i]))
        #np.append(XL[i], X[1][i])
        #np.append(XL[i], X[2][i])
        #np.append(XL[i], X[3][i])
    print(len(XL2[0]))
    '''
    #XL = PCA(n_components = 10).fit_transform(XL)

    #XL = features.extractFeaturesWithMask(
            #X, channelIndex, featuremask=[0,1,2,3,4,5,6,7,9,10,12,13,15,17,18,19,20,21,22,23,25,26], printTime=False)
    #XLreturn = features.extractFeaturesWithMask(Xreturn, channelIndex, featuremask=[0,1,2,3,4,5,6], printTime=True)


    #Scale the data if needed and split dataset into training and testing

    #Migth be an idea to use the same scaler, see classifier.py file
    #scaler = classifier.makeScaler(joinedXL)
    # scaledData = classifier.realTimescale(XL, scaler)
    # XLtrain, XLtest, yTrain, yTest = split(XL, y)

    #proposed solution, change input to makeScaler when creating for one subject. See classifier.py
    #XLjoined = np.append(XL1, XL2, axis = 0)
    scaler = classifier.makeScaler(XL2)

    XLtrain1, XLtest1, yTrain1, yTest1, XL1 = classifier.scaleAndSplit(XL1, y1[0], scaler)
    XLtrain2, XLtest2, yTrain2, yTest2, XL2 = classifier.scaleAndSplit(XL2, y2[0], scaler)


    #XLtrain1T, XLtest1T, yTrain1T, yTest1T, XL1T = classifier.scaleAndSplit(XL1T, y1T[0], scaler)
    #XLtrain2T, XLtest2T, yTrain2T, yTest2T, XL2T = classifier.scaleAndSplit(XL2T, y2T[0], scaler)

    #This is to combine the two subjects

    #yTrain = np.append(yTrain1, yTrain2, axis = 0)
    #XLtrain = np.append(XLtrain1, XLtrain2, axis = 0)
    #yTest = np.append(yTest1, yTest2, axis = 0)
    #XLtest = np.append(XLtest1, XLtest2, axis = 0)


    #bestParams.append(classifier.tuneSvmParameters(XLtrain, yTrain, XLtest, yTest, n_jobs = -1))
    #bestParams.append(tuneDecisionTreeParameters(XLtrain, yTrain, XLtest, yTest, n_jobs = -1))


    #try this with tuning of parameters later today.

    #clf, clfPlot = createAndTrain(XLtrain, yTrain, bestParams[0])



    #Use this if predictor other than SVM is used.
    clf, clfPlot = createAndTrain(XLtrain2, yTrain2, None)
    #plot.trainingPredictions(clf, XL, y[0])

    ###TO PLOT LEARNING CURVE UNCOMMENT THIS.
    #title = "Learning Curves (SVM, RBF kernel, C = 50, $\gamma=0.001$)"
    #estimator = svm.SVC(kernel = 'rbf', gamma = 0.01, C = 50, decision_function_shape = 'ovr')
    #plot.learningCurve(estimator, title, XL, y[0], (0.7, 1.01), cv=20, n_jobs=-1)
    #plt.show()

    #clf = classifier.loadMachineState(classifierstring)
    classifier.saveMachinestate(clf, classifierstring)   #Uncomment this to save the machine state
    classifier.saveScaler(scaler, classifierstring)
    #clf = CalibratedClassifierCV(svm.SVC(kernel = 'linear', C = C, decision_function_shape = 'ovr'), cv=5, method='sigmoid')

    #Use this if it is important to see the overall prediction, and not for only the test set


    scores = cross_val_score(clf, XLtrain2, yTrain2, cv=50, scoring = 'accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()
    print("Scores")
    print(scores)



    tempAccuracyScore, tempPrecision, tempClassificationReport, tempf1Score = classifier.predictTimer(XLtest2, clf, yTest2)
    accuracyScore.append(tempAccuracyScore)
    f1Score.append(tempf1Score)
    precision.append(tempPrecision)
    classificationReport.append(tempClassificationReport)

    #crossValScore.append(tempCrossValScore)
    #accuracyScore, classificationReport = compareFeatures(XL, XLtrain, yTrain, XLtest, yTest, bestParams)

    print()
    print("The best parameters for the different channels are:")
    print()
    print(bestParams)
    print()
    print("The prediction accuracy for the different channels is:")
    print(accuracyScore)
    print("The f1 score which include false negatives etc is:")
    print(f1Score)#This score says something about the correctness of the prediction.
    print("The precision score:")
    print(precision)#This score says something about the correctness of the prediction.
    print("Classification Report of channel %d:" %channelIndex) #String, weird if you print whole array with string, and predicting over several channels.
    print(classificationReport[0])

    #evaluateFeatures(XLtrain, yTrain)

    #print("The cross-validation score is:")
    #print(crossValScore)
    #prints the classification report

    #for i in range(len(classificationReport)):
        #print "Number of features :", i + 1
        #print(classificationReport[i])


    #plotClassifier(clfPlot, XLtrain) #Not sure if this works atm. Uncomment this to plot the classifier









def createAndTrain(XLtrain, yTrain, bestParams):
    #preprocessing.scale might need to do this scaling, also have to tune the classifier parameters in that case

    print("Starting to train the classifier")

    print()


    #This fits with the tested best parameters. Might want to manually write this
    #if bestParams['kernel'] == 'linear':
        #clf = svm.SVC(kernel =bestParams['kernel'], C = bestParams['C'], decision_function_shape = 'ovr')
    #else:
        #clf = svm.SVC(kernel = bestParams['kernel'], gamma=bestParams['gamma'], C= bestParams['C'], decision_function_shape='ovr')

    C = 10
    #C = 50
    clf = svm.SVC(kernel = 'rbf', gamma = 0.01, C = C, decision_function_shape = 'ovr')
    #clf = svm.LinearSVC(penalty = 'l2',  loss='squared_hinge', dual = False, C = 10, random_state = 42)
    #clf = svm.SVC(kernel = 'linear', C = C, decision_function_shape = 'ovr')
    #clf = linear_model.SGDClassifier(penalty = 'l2', random_state = 42)


    #Decision tree classification. max_depth 8 and min_leaf = 5 gives good results, but varying.
    #clf = tree.DecisionTreeClassifier(max_depth = None, min_samples_leaf=5)
    #clf = tree.DecisionTreeClassifier(max_depth = None, min_samples_leaf = bestParams['min_samples_leaf'])
    #randomForestClassifier.
    #clf = RandomForestClassifier(n_estimators = 54, max_depth = 30,  min_samples_leaf = 1, random_state = 40)
    #clf = RandomForestClassifier(max_depth = bestParams['max_depth'], min_samples_leaf = bestParams['min_samples_leaf'], n_estimators = bestParams['n_estimators'], random_state = 40)

    #clf = neighbors.KNeighborsClassifier(n_neighbors = 5, n_jobs = -1)
    cost = []
    for i in range(500):
        print(i)
        start = time.time()
        clf.fit(XLtrain,yTrain)#skaler???
        cost.append(time.time() - start)
    #Create classifier to be able to visualize it
    #clfPlot = clf
    #clfPlot.fit(XLtrain,yTrain)
    print("Time taken to train classifier:")
    print(min(cost))

    return clf, None #clfPlot Uncomment this to be able to plot the classifier



if __name__ == '__main__':
	main()
