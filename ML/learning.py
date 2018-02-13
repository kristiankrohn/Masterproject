import pyeeg
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from numpy.fft import fft
from numpy import zeros, floor
import math
import time

import dataset
from globalconst import  *
import globalvar
import copy


yTestGUI = []
predictionsGUI = []





def main():
    partialPathZ = "c:\Users\Adrian Ribe\Desktop\Masteroppgave\Code\Machine learning trial\Z\Z"
    partialPathO = "c:\Users\Adrian Ribe\Desktop\Masteroppgave\Code\Machine learning trial\O\O"
    extractFeatures(partialPathZ, partialPathO)





def startLearning():
    bestParams = []
    accuracyScore = []
    f1Score = []
    precision = []
    classificationReport = []
    #crossValScore = []
    X, y = dataset.loadDataset("longdata.txt")

    #Length = how many examples of each class is desired.
    X, y = dataset.sortDataset(X, y, length=100000, classes=[0,5,6,4,2,8]) #,6,4,2,8

    #def sortDataset(x=None, y=None, length=10, classes=[0,5,4,2,6,8])
    #if x or y is undefined, data.txt will be loaded


    channelIndex = 0
    for channel in range(1): #len(X) for aa ha med alle kanaler
        XL = extractFeatures(X, channelIndex)
    #XL = extractFeatures(X)



        #Trying something new with cross_val_score

        #Scale the data if needed and split dataset into training and testing
        XLtrain, XLtest, yTrain, yTest = scaleAndSplit(XL, y[0])

        #trenger en losning for aa ha samme split hver gang
        #XLscaled, XLtrain, XLtest, yTrain, yTest = scaleAndSplit(XL, y[0])
        #bestParams.append(tuneSvmParameters(XLtrain, yTrain, XLtest, yTest))
        #bestParams.append(tuneDecisionTreeParameters(XLtrain, yTrain, XLtest, yTest))

        #scaler = StandardScaler()

        #pca = PCA(n_components = 2)
        #XLtrain = scaler.fit_transform(XLtrain, yTrain)
        #XLtest = scaler.fit_transform(XLtest, yTest)
        #pca.fit_transform(XLtrain, yTrain)
        #pca.fit_transform(XLtest, yTest)
        #try this with tuning of parameters later today.

        #clf, clfPlot = createAndTrain(XLtrain, yTrain, bestParams[channel])



        #Use this if predictor other than SVM is used.
        clf, clfPlot = createAndTrain(XLtrain, yTrain, None)
        #clf = loadMachineState("learningState99HFDreducedSvm")
        #saveMachinestate(clf, "learningState99HFDslopedSVM")   #Uncomment this to save the machine state

        #Use this if it is imporatnt to see the overall prediction, and not for only the test set
        scores = cross_val_score(clf, XLtrain, yTrain, cv=5, scoring = 'precision_macro')
        print("Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print()
        #compareFeatures(XL, XLtrain, yTrain, XLtest, yTest)

        tempAccuracyScore, tempClassificationReport, tempf1Score, tempPrecision = predict(XLtest, clf, yTest)
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

def scaleAndSplit(XL, labels):
    XLscaled = (np.array(XL))
    XLtrain, XLtest, yTrain, yTest = train_test_split(XLscaled, labels, test_size = 0.2, random_state = 42, stratify = labels)

    return XLtrain, XLtest, yTrain, yTest

def extractFeatures(X, channel):
    XL = [[]]
    frequencyBands = [0.1, 4, 8, 12,30]
    Fs = 250
    featureVector = []

    for i in range(len(X[0])):
        startTime = time.time()
        power, powerRatio = pyeeg.bin_power(X[channel][i], frequencyBands, Fs)
        bandAvgAmplitudesCh1 = getBandAmplitudes(X[0][i], frequencyBands, Fs)
        bandAvgAmplitudesCh2 = getBandAmplitudes(X[1][i], frequencyBands, Fs)
        bandAvgAmplitudesCh3 = getBandAmplitudes(X[0][i], frequencyBands, Fs)
        bandAvgAmplitudesCh4 = getBandAmplitudes(X[1][i], frequencyBands, Fs)
        thetaBetaRatioCh1 = bandAvgAmplitudesCh1[1]/bandAvgAmplitudesCh1[3]
        thetaBetaRatioCh2 = bandAvgAmplitudesCh2[1]/bandAvgAmplitudesCh2[3]
        thetaBetaRatioCh3 = bandAvgAmplitudesCh3[1]/bandAvgAmplitudesCh3[3]
        thetaBetaRatioCh4 = bandAvgAmplitudesCh4[1]/bandAvgAmplitudesCh4[3]

        pearsonCoefficients13 = np.corrcoef(X[0][i], X[2][i])
        pearsonCoefficients14 = np.corrcoef(X[0][i], X[3][i])
        pearsonCoefficients23 = np.corrcoef(X[1][i], X[2][i])
        pearsonCoefficients24 = np.corrcoef(X[1][i], X[3][i])
        cov34 = np.cov(list(X[2][i]), list(X[3][i]))
        cov12 = np.cov(list(X[0][i]), list(X[1][i]))
        cov34 = np.cov(list(X[2][i]), list(X[3][i]))
        corr12 = np.correlate(list(X[0][i]), list(X[1][i])),

        maxIndex = np.argmax(list(X[channel][i]))
        minIndex = np.argmin(list(X[channel][i]))
        minValueCh1 = np.amin(list(X[0][i]))
        maxValueCh1 = np.amax(list(X[0][i]))
        slopeCh1 = (minValueCh1 - maxValueCh1)/ (minIndex - maxIndex)
        #print(power)
        #print(channel)
        #thetaBetaPowerRatio = power[1]/power[3] denne sugde tror jeg
        #featureVector = [power[1], pyeeg.hurst(list(X[0][i])), np.std(list(X[0][i])),  np.ptp(list(X[0][i])), np.amax(list(X[0][i])), np.amin(list(X[0][i]))]
        featureVector = [
                        pyeeg.hfd(list(X[channel][i]), 200), #Okende tall gir viktigere feature, men mye lenger computation time
                        np.amin(list(X[0][i])) - np.amin(list(X[2][i])),
                        np.amax(list(X[0][i])) - np.amax(list(X[2][i])),
                        pyeeg.spectral_entropy(list(X[channel][i]), [0.1, 4, 7, 12,30], 250, powerRatio),
                        #pearsonCoefficients14[0][1],
                        pearsonCoefficients14[1][0],
                        np.std(list(X[channel][i])),
                        slopeCh1,
                        #np.amax(list(X[0][i])),
                        #np.amax(list(X[1][i])),
                        #np.amin(list(X[0][i])),
                        #np.amin(list(X[1][i])),

                        #thetaBetaRatioCh1,
                        #thetaBetaRatioCh2,
                        #thetaBetaRatioCh3,
                        #thetaBetaRatioCh4,
                        #np.ptp(list(X[0][i])), #denne er bra
                        #powerRatio[2],
                        #powerRatio[1],




                        #np.amin(list(X[0][i])),


                        #corr12[0][0],
                        #np.amax(list(X[0][i])) - np.amax(list(X[3][i])),
                        #np.correlate(list(X[0][i]), list(X[2][i])),

                        #cov12[0][1],
                        #cov12[1][0],
                        #np.amax(list(X[0][i])) - np.amax(list(X[2][i])),

                        #np.amax(list(X[1][i])) - np.amax(list(X[2][i])),
                        #np.amax(list(X[1][i])) - np.amax(list(X[3][i])),
                        #pearsonCoefficients13[0][1],
                        #pearsonCoefficients13[1][0],
                        #thetaBetaRatio,

                        #powerRatio[0],
                        #pyeeg.hurst(list(X[channel][i])),
                        #np.std(list(X[channel][i])),
                        #np.var(list(X[channel][i])),

                        #np.correlate(list(X[0][i]), list(X[3][i])), #funker bare for decision tree???
                        #np.correlate(list(X[2][i]), list(X[3][i])),

                        #cov34[0][1],
                        #cov34[1][0],

                        #bandAvgAmplitudes[0],
                        #bandAvgAmplitudes[1],
                        #np.std(list(X[channel][i])),
                        #pyeeg.hjorth(list(X[0][i])),


                        #np.ptp(list(X[3][i])),





                        #thetaBetaPowerRatio,
                        #powerRatio[1],
                        #powerRatio[2],
                        #powerRatio[3],
                        #power[0],
                        #power[1],
                        #power[2],
                        #power[3],
                        #pyeeg.pfd(list(X[0][i])),
                        #pyeeg.dfa(list(X[0][i]), None, None),
                        ]
        XL.append(featureVector)
        print("Time taken to extract features for example %d: " % i)
        print(time.time() - startTime)
        #print(XL)
    XL.pop(0)
    return XL


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
    C = 10000
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

def tuneSvmParameters(XLtrain, yTrain, XLtest, yTest):
    bestParams = []
    tunedParameters = [{'kernel': ['rbf'], 'gamma': [1e0, 1e-1, 1e-2, 1e-3, 1e-4],
                        'C': [1, 10, 50, 100, 500, 1000, 10000]},
                        {'kernel': ['linear'], 'C': [1, 10, 50, 100, 500, 1000, 10000]}]

    scores = ['precision', 'recall']

    #for score in scores:

    #CV = ?????Increase CV = less variation, more computation time
    #comment in for loop and add score where scores[0] is atm, to make best parameters for prediction and recall
    print("Starting to tune parameters")
    print()

    clf = GridSearchCV(svm.SVC(), tunedParameters, cv=10, scoring='%s_macro' % scores[0])
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

    #This loop prints out the mean from all the 10 fold combinations standarddeviation and lots of good stuff

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

def compareFeatures(XL, XLtrain, yTrain, XLtest, yTest):
    accuracyScore = []
    f1Score = []
    precisionScore = []
    #Nested loops to compare accuracy when varying number of features are used.
    for i in range(len(XL[0])):
        compFeaturesTraining, compFeaturesTest = appendFeaturesForComparison(i, XL, XLtrain, XLtest)
        #Create the classifier and train it on the test data.
        clf, clfPlot = createAndTrain(compFeaturesTraining, yTrain, None) #uncomment this if state should be loaded

        #Load state of the classifier
        #clf = loadMachineState() #Uncomment this to load the machine state

        #Predict the classes
        tempAccuracy, tempClassReport, tempf1Score, tempPrecision = predict(compFeaturesTest, clf, yTest)
        print("Accuracy f1 and preision for %d features: " %i)





        accuracyScore.append(tempAccuracy)
        f1Score.append(tempf1Score)
        precisionScore.append(tempPrecision)
        #crossValScore.append(tempCrossValScore)
    print("Feature comparison recap: ")
    print()
    print("Accuracy:")
    print(accuracyScore)
    print("F1 score:")
    print(f1Score)
    print("Precision")
    print(precisionScore)

    return accuracyScore, f1Score, precisionScore

def predict(Xtest, clf, yTest):
    print("Starting to predict")
    start = time.time()
    predictions = clf.predict(Xtest)
    print("Time taken to predict with given examples:")
    print(time.time() - start)
    #Print the test data to see how well it performs.
    print(yTest)
    print(predictions)
    #print("HALLO", clf.predict_proba(Xtest))
    accuracyScore = accuracy_score(yTest, predictions)
    precision = precision_score(yTest, predictions, average = 'macro')
    print("Accuracy score:")
    print(accuracyScore)
    print("Precision score:")
    print(precision)
    #print(accuracyScore)
    #meanSquaredScore = mean_squared_error(yTest, predictions)
    classificationReport = classification_report(yTest, predictions)
    f1Score = f1_score(yTest, predictions, average='macro')
    #crossValScore = cross_val_score(clf, Xtest, yTest, cv=2)
    #print(classificationReport)
    #print(meanSquaredScore)

    return accuracyScore, classificationReport, f1Score, precision

def predictGUI(X, clf, y, windowLength):
    global yTestGUI, predictionsGUI
    yTest = [y]
    #print("Starting to predict with GUI")
    start = time.time()
    X = dataset.shapeArray(X, windowLength, y)
    if X == -1:
        return

    Xtest = extractFeatures(X, 0)
    predictions = clf.predict(Xtest)
    print("Time taken to predict with given examples in GUI:")
    print(time.time() - start)
    #Print the test data to see how well it performs.
    if yTest == predictions:
        print("Correct prediction of %d!" %predictions)
    else:
        print("Should have predicted:")
        print(yTest)
        print()
        print("Actually predicted:")
        print(predictions)
    yTestGUI.append(y)
    predictionsGUI.append(predictions[0])
    #print("HALLO", clf.predict_proba(Xtest))
    #accuracyScoreGUI = accuracy_score(yTest, predictions)
    #precisionGUI = precision_score(yTest, predictions, average = 'macro')
        #print("Accuracy score:")
        #print(accuracyScore)
        #print("Precision score:")
        #print(precision)
    #print(accuracyScore)
    #meanSquaredScore = mean_squared_error(yTest, predictions)
        #classificationReportGUI = classification_report(yTest, predictions)
        #f1Score = f1_score(yTest, predictions, average='macro')
    #crossValScore = cross_val_score(clf, Xtest, yTest, cv=2)
    #print(classificationReport)
    #print(meanSquaredScore)
def classificationReportGUI():
    global yTestGUI, predictionsGUI

    accuracyScore = accuracy_score(yTestGUI, predictionsGUI)
    precision = precision_score(yTestGUI, predictionsGUI, average = 'macro')
    classificationReport = classification_report(yTestGUI, predictionsGUI)
    f1Score = f1_score(yTestGUI, predictionsGUI, average='macro')
    print("Should have predicted: ")
    print(yTestGUI)
    print("Actually predicted: ")
    print(predictionsGUI)
    print("Accuracy score:")
    print(accuracyScore)
    print("Precision score:")
    print(precision)
    print("Classification report:")
    print(classificationReport)



def predictRealTime(X, clf):
    Xtest = extractFeatures(X, 0)

    print("Starting to predict")
    start = time.time()
    predictions = clf.predict(Xtest)
    timeStop = time.time()
    print("The prediction is:")
    print(predictions)
    print()
    print("Time taken to predict with given examples:")
    print(timeStop - start)


    #Print the test data to see how well it performs.

    #accuracyScore = accuracy_score(yTest, predictions)
    #precision = precision_score(yTest, predictions, average = 'macro')
    #print("Accuracy score:")
    #print(accuracyScore)
    #print("Precision score:")
    #print(precision)

    #meanSquaredScore = mean_squared_error(yTest, predictions)
    #classificationReport = classification_report(yTest, predictions)
    #f1Score = f1_score(yTest, predictions, average='macro')

def saveMachinestate(clf, string):
    joblib.dump(clf, "ML\\" + string + ".pkl")

def loadMachineState(string):

    clf = joblib.load("ML\\" + string + ".pkl")
    return clf

def getBandAmplitudes(X, Band, Fs):
    C = fft(X)
    C = abs(C)
    avgAmplitude =zeros(len(Band)-1);

    for Freq_Index in range(len(Band)-1):
        Freq = float(Band[Freq_Index])										## Xin Liu
        Next_Freq = float(Band[Freq_Index+1])
		#Endret til Int for aa faa det til aa gaa igjennom
        avgAmplitude[Freq_Index] = sum(C[int(Freq/Fs*len(X)):int(Next_Freq/Fs*len(X))]) / len(C[int(Freq/Fs*len(X)):int(Next_Freq/Fs*len(X))])
    return avgAmplitude



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

def decisionTreeInfo(XLtrain, yTrain, XLtest):
    estimator = tree.DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    estimator.fit(XLtrain, yTrain)

    # The decision estimator has an attribute called tree_  which stores the entire
    # tree structure and allows access to low level attributes. The binary tree
    # tree_ is represented as a number of parallel arrays. The i-th element of each
    # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    # Some of the arrays only apply to either leaves or split nodes, resp. In this
    # case the values of nodes of the other type are arbitrary!
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node
    #

    # Using those arrays, we can parse the tree structure:

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = estimator.decision_path(XLtest)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(XLtest)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        if leave_id[sample_id] != node_id:
            continue

        if (XLtest[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (XLtest[%s, %s] (= %s) %s %s)"
              % (node_id,
                 sample_id,
                 feature[node_id],
                 XLtest[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))

    # For a group of samples, we have the following common node.
    sample_ids = [0, 1,2,3,4,5,6,7,8,9,10,11,12]
    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                    len(sample_ids))

    common_node_id = np.arange(n_nodes)[common_nodes]

    print("\nThe following samples %s share the node %s in the tree"
          % (sample_ids, common_node_id))
    print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))

if __name__ == '__main__':
	main()
