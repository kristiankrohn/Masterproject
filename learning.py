import pyeeg
import numpy as np
from sklearn import svm
from sklearn import preprocessing
import itertools
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV

from sklearn.preprocessing import StandardScaler
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
from sklearn.feature_selection import SelectFromModel
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
    startLearning()


def startLearning():
    bestParams = []
    accuracyScore = []
    f1Score = []
    precision = []
    classificationReport = []
    #crossValScore = []
    #X, y = dataset.loadDataset("longdata.txt")
    X, y = dataset.loadDataset("data.txt")
    y = dataset.mergeLabels(y)
    X, y = dataset.sortDataset(X, y, length=100000, classes=[0,5,6,4,2,8]) #,6,4,2,8

    #def sortDataset(x=None, y=None, length=10, classes=[0,5,4,2,6,8])
    #if x or y is undefined, data.txt will be loaded


    channelIndex = 0
    '''
    FUNC_MAP = {0: hfd, 
            1: minDiff, 
            2: maxDiff,
            3: specEntropy,
            4: pearsonCoeff,
            5: stdDeviation,
            6: slope,
            7: thetaBeta1,
            8: extrema}
    '''
    #XL = features.extractFeatures(X, channelIndex)
    XL = features.extractFeaturesWithMask(X, channelIndex, featuremask=[0,1,2,3,4,5,6], printTime=True)
    #Scale the data if needed and split dataset into training and testing
    XLtrain, XLtest, yTrain, yTest = classifier.scaleAndSplit(XL, y[0])

    #bestParams.append(tuneSvmParameters(XLtrain, yTrain, XLtest, yTest))
    #bestParams.append(tuneDecisionTreeParameters(XLtrain, yTrain, XLtest, yTest))

    scaler = StandardScaler()

    
    XLtrain = scaler.fit_transform(XLtrain, yTrain)
    XLtest = scaler.fit_transform(XLtest, yTest)
    
    
    #try this with tuning of parameters later today.

    #clf, clfPlot = createAndTrain(XLtrain, yTrain, bestParams[channel])



    #Use this if predictor other than SVM is used.
    clf, clfPlot = createAndTrain(XLtrain, yTrain, None)
    #clf = classifier.loadMachineState("learningState99HFDslopedSVM") 
    #classifier.saveMachinestate(clf, "learningState99HFDslopedSVM")   #Uncomment this to save the machine state
    #clf = CalibratedClassifierCV(svm.SVC(kernel = 'linear', C = C, decision_function_shape = 'ovr'), cv=5, method='sigmoid')

    #Use this if it is imporatnt to see the overall prediction, and not for only the test set
    #scores = cross_val_score(clf, XLtrain, yTrain, cv=5, scoring = 'precision_macro')
    #print("Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #print()
    #print("Scores")
    #print(scores)

    tempAccuracyScore, tempPrecision, tempClassificationReport, tempf1Score = classifier.predict(XLtest, clf, yTest)
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

    evaluateFeatures(XLtrain, yTrain)

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
    start = time.time()
    print()


    #This fits with the tested best parameters. Might want to manually write this
    #if bestParams['kernel'] == 'linear':
    #    clf = svm.SVC(kernel =bestParams['kernel'], C = bestParams['C'], decision_function_shape = 'ovr')
    #else:
    #    clf = svm.SVC(kernel = bestParams['kernel'], gamma=bestParams['gamma'], C= bestParams['C'], decision_function_shape='ovr')
    #C = 10000
    C = 500
    #clf = svm.SVC(kernel = 'rbf, gamma = 0.12, C = C, decision_function_shape = 'ovr')
    clf = svm.SVC(kernel = 'linear', C = C, decision_function_shape = 'ovr')

    #Decision tree classification. max_depth 8 and min_leaf = 5 gives good results, but varying.
    #clf = tree.DecisionTreeClassifier(max_depth = None, min_samples_leaf=5)
    #clf = tree.DecisionTreeClassifier(max_depth = None, min_samples_leaf = bestParams['min_samples_leaf'])
    #randomForestClassifier.
    #clf = RandomForestClassifier(max_depth = None,  min_samples_leaf = 10, random_state = 40)
    #clf = RandomForestClassifier(max_depth = None, min_samples_leaf = bestParams['min_samples_leaf'], random_state = 40)

    #clf = neighbors.KNeighborsClassifier(5)
    clf.fit(XLtrain,yTrain)#skaler???

    #Create classifier to be able to visualize it
    #clfPlot = clf
    #clfPlot.fit(XLtrain,yTrain)
    print("Time taken to train classifier:")
    print(time.time() - start)
    return clf, None #clfPlot Uncomment this to be able to plot the classifier



def tuneDecisionTreeParameters(XLtrain, yTrain, XLtest, yTest):
    bestParams = []

    sampleLeafPipeline = [{'min_samples_leaf': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 500]}]

    print("Starting to tune parameters")
    print()

    clf = GridSearchCV(tree.DecisionTreeClassifier(), sampleLeafPipeline, cv=10, scoring='%s_macro' % 'precision')

    clf.fit(XLtrain, yTrain)
    bestParams.append(clf.best_params_)
    print("Best parameters set found on development set:")
    print()
    print(bestParams)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    #This loop prints out standarddeviation and lots of good stuff

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    yPred = clf.predict(XLtest)
    print(classification_report(yTest, yPred)) #ta denne en tab inn for aa faa den tilbake til original
    print()

    return bestParams[0]

def evaluateFeatures(X, y):

    #Make all the different models and train them to find the importance of the features
    forest = ExtraTreesClassifier(n_estimators = 250, random_state = 0)
    forest.fit(X,y)
    #Print to see what the shape looks like before features are removed
    print(X.shape)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    #If this is used, the worst performing feature will be removed
    model = SelectFromModel(forest, prefit=True) #This removes the worst features.
    Xnew = model.transform(X)
    #Print to check that feature has been removed
    #print(Xnew.shape)
    #return Xnew









def appendFeaturesForComparison(i, XL, XLtrain, XLtest):

    compareFeaturesTraining = []
    compareFeaturesTesting = []
    for j in range(len(XLtrain)):
        compareFeaturesTraining.append(list(XLtrain[j][0:(i + 1)]))
    for j in range(len(XLtest)):
        compareFeaturesTesting.append(list(XLtest[j][0:(i + 1)]))
    return compareFeaturesTraining, compareFeaturesTesting


def plotClassifier(clf, XLtrain):
    title = 'SVC with RBF-kernel'

    Xtrain = np.array(XLtrain)

    # Set-up 2x2 grid for plotting.
    fig, ax = plt.subplots()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    X0, X1 = Xtrain[:, 0], Xtrain[:, 1]
    xx, yy = makeMeshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('DFA')
    ax.set_ylabel('STD')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)


    plt.show()

def makeMeshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out








if __name__ == '__main__':
	main()
