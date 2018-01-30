import pyeeg
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

import dataset
from globalconst import  *
import globalvar
import copy



dataPoints = [[]]

labels = []
#features
dfaOpen = []
hurstOpen = []
pfdOpen = []
stdOpen = []

dataSetOpen = []
dataSetClosed = []

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

    extractFeatures(partialPathZ, partialPathO)





def startLearning():
    XL = [[]]

    accuracyScore = []
    classificationReport = []
    featureVector = []

    X, y = dataset.loadDataset("longtemp.txt")
    for i in range(len(X[0])):
        power, powerRatio = pyeeg.bin_power(X[0][i], [0.1, 4, 7, 12,30], 250)
        #featureVector = [power[1], pyeeg.hurst(list(X[0][i])), np.std(list(X[0][i])),  np.ptp(list(X[0][i])), np.amax(list(X[0][i])), np.amin(list(X[0][i]))]
        featureVector = [powerRatio[0],
                        pyeeg.hurst(list(X[0][i])),
                        #powerRatio[1],
                        #powerRatio[2],
                        #powerRatio[3],
                        #power[0],
                        #power[1],
                        #power[2],
                        #power[3],
                        #pyeeg.pfd(list(X[0][i])),
                        np.std(list(X[0][i])),
                        pyeeg.hfd(list(X[0][i]), 200), #denne maa testes med forskjellige Kverdier, vet ikke hva den betyr
                        #pyeeg.hjorth(list(X[0][i])),
                        pyeeg.spectral_entropy(list(X[0][i]), [0.1, 4, 7, 12,30], 250, powerRatio),
                        #pyeeg.dfa(list(X[0][i]), None, None),
                        #np.ptp(list(X[0][i])),
                        #np.amax(list(X[0][i])),
                        #np.amin(list(X[0][i]))
                        ]
        #, np.var(list(X[0][i])),
        #, np.max(list(X[0][i]))
        #, np.min(list(X[0][i]))


        #Suggested doing the below if a vector should be added as feature
        #cov = np.cov(list(X[0][i]), list(X[6][i]))
        #cov = np.ravel(cov)
        #featureVector = np.concatenate((featureVector, cov))
        #print(cov)
        #pyeeg.dfa(list(X[0][i]))] #pyeeg.dfa(list(X[0][i])
        #pyeeg.pfd(list(X[0][i])) denne gjor det sykt daarlig med resten


        XL.append(featureVector)
    XL.pop(0)

    #Scale the data if needed and split dataset into training and testing
    XLscaled, XLtrain, XLtest, yTrain, yTest = scaleAndSplit(XL, y[0])


    #Nested loops to compare accuracy when varying number of features are used.
    for i in range(len(XL[0])):
        compFeaturesTraining, compFeaturesTest = compareFeatures(i, XL, XLtrain, XLtest)
        #Create the classifier and train it on the test data.
        clf, clfPlot = createAndTrain(compFeaturesTraining, yTrain) #uncomment this if state should be loaded

        #Load state of the classifier
        #clf = loadMachineState() #Uncomment this to load the machine state

        #Predict the classes
        tempAccuracy, tempClassReport = predict(compFeaturesTest, clf, yTest)

        accuracyScore.append(tempAccuracy)
        classificationReport.append(tempClassReport)
    #Evaluate the features on the training data.
    evaluateFeatures(XLtrain, yTrain)
    print(accuracyScore)

    #prints the classification report

    #for i in range(len(classificationReport)):
        #print "Number of features :", i + 1
        #print(classificationReport[i])

    #saveMachinestate(clf)   #Uncomment this to save the machine state
    #plotClassifier(clfPlot) #uncomment this to plot the classifier

def scaleAndSplit(dataPoints, labels):

    XLscaled = (np.array(dataPoints))
    XLtrain, XLtest, yTrain, yTest = train_test_split(XLscaled, labels, test_size = 0.1)

    return XLscaled, XLtrain, XLtest, yTrain, yTest

def compareFeatures(i, XL, XLtrain, XLtest):

    compareFeaturesTraining = []
    compareFeaturesTesting = []
    for j in range(len(XLtrain)):
        compareFeaturesTraining.append(list(XLtrain[j][0:(i + 1)]))
    for j in range(len(XLtest)):
        compareFeaturesTesting.append(list(XLtest[j][0:(i + 1)]))
    return compareFeaturesTraining, compareFeaturesTesting

def createAndTrain(XLtrain, yTrain):
    #preprocessing.scale might need to do this scaling, also have to tune the classifier parameters in that case

    #SVM classification, regulation parameter C = 102 gives good results
    #C = 102
    #clf = svm.SVC(kernel='rbf', gamma=0.12, C=C, decision_function_shape='ovr')

    #Decision tree classification. max_depth 8 and min_leaf = 5 gives good results, but varying.
    clf = tree.DecisionTreeClassifier(max_depth = 8, min_samples_leaf=5)
    clf.fit(XLtrain,yTrain)#skaler???

    #Create classifier to be able to visualize it
    clfPlot = clf
    clfPlot.fit(XLtrain,yTrain)
    return clf, clfPlot #Uncomment this to be able to plot the classifier

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
    print(Xnew.shape)
    #return Xnew


def predict(Xtest, clf, yTest):
    predictions = clf.predict(Xtest)
    #Print the test data to see how well it performs.
    print(yTest)
    print(predictions)
    accuracyScore = accuracy_score(yTest, predictions)
    print(accuracyScore)
    #meanSquaredScore = mean_squared_error(yTest, predictions)
    classificationReport = classification_report(yTest, predictions)
    #print(classificationReport)
    #print(meanSquaredScore)

    return accuracyScore, classificationReport


def saveMachinestate(clf):
    joblib.dump(clf, 'learningState.pkl')

def loadMachineState():
    clf = joblib.load('learningState.pkl')
    return clf

def plotClassifier(clf):
    global dataPoints
    title = 'SVC with RBF-kernel'

    Xtrain = np.array(dataPoints)

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
