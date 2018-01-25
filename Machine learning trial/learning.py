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
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
import graphviz

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
#To read and extract features from eyes open

def extractFeatures(partialPathZ, partialPathO):
    for i in range(100):
        index = "{:03}".format(i + 1)
        filepath = partialPathZ + str(index) + ".txt"
        i+= 1
        file = open(filepath, 'r')
        tmp = file.readlines()
        dataPoint = [float(k) for k in tmp]
        dataSetOpen.append(dataPoint)
        dfaOpen.append(pyeeg.dfa(dataPoint))
        #hurstOpen.append(pyeeg.hurst(dataPoint))
        pfdOpen.append(pyeeg.pfd(dataPoint))
        stdOpen.append(np.std(dataPoint))
        #specEntropyOpen.append(pyeeg.spectral_entropy(dataPoint))

#To read and extract features from eyes closed
    for i in range(100):
        index = "{:03}".format(i + 1)
        filepath = partialPathO + str(index) + ".txt"
        i+= 1
        file = open(filepath, 'r')
        tmp = file.readlines()
        dataPoint = [float(k) for k in tmp]
        dataSetClosed.append(dataPoint)
        dfaClosed.append(pyeeg.dfa(dataPoint))
        #hurstClosed.append(pyeeg.hurst(dataPoint))
        pfdClosed.append(pyeeg.pfd(dataPoint))
        stdClosed.append(np.std(dataPoint))
        #specEntropyClosed.append(pyeeg.spectral_entropy(dataPoint))

    createDataset()
    #exportplot(dataSetOpen, dataSetClosed, "Eyes Open", "Eyes Closed") # Uncomment this to export plots into folder
    #print(labels)
    clf, clfPlot = createAndTrain()
    #saveMachinestate(clf)   #Uncomment this to save the machine state
    #plotClassifier(clfPlot) #uncomment this to plot the classifier

def createDataset():
    global dfaOpen, dfaClosed, pfdOpen, pfdClosed, labels, dataPoints
    featureVector = []

    for i in range(len(dfaOpen)):
        featureVector = [dfaOpen[i], stdOpen[i], pfdOpen[i]]
        dataPoints.append(featureVector)
        labels.append(0)
        featureVector = [dfaClosed[i], stdClosed[i], pfdOpen[i]]
        dataPoints.append(featureVector)
        labels.append(1)
    dataPoints.pop(0) #remove empty list at start position



def exportplot(eyesOpenData, eyesClosedData,  titleOpen="", titleClosed = "", ax=None):
    if ax == None:
        fig, (ax, ax1) = plt.subplots(2)

    length = len(eyesOpenData)
    x = np.arange(0, length/250.0, 1.0/250.0)
    ax.set_autoscaley_on(False)
    ax.set_ylim([-200,200])
    ax1.set_autoscaley_on(False)
    ax1.set_ylim([-200,200])

    ax.plot(x, eyesOpenData, label=titleOpen)
    ax.set_title(titleOpen)
    ax1.plot(x, eyesClosedData, label =titleClosed  )
    ax1.set_title(titleClosed)
    plt.ylabel('uV')
    plt.xlabel('Seconds')
    plt.show()

def createAndTrain():
    global dataPoints, labels
    #preprocessing.scale might need to do this scaling, also have to tune the classifier parameters in that case
    Xscaled = (np.array(dataPoints))
    Xtrain, Xtest, yTrain, yTest = train_test_split(Xscaled, labels, test_size = 0.1)

    evaluateFeatures(Xtrain, yTrain)


    C = 102 #SVM regulation parameter, 100 gives good results
    clf = svm.SVC(kernel='rbf', gamma=0.12, C=C, decision_function_shape='ovr')

    #clf = tree.DecisionTreeClassifier(max_depth = 8, min_samples_leaf=5) #max_depth 8 and min_leaf = 5 gives good results, but varying.
    clf.fit(Xtrain,yTrain)#skaler???

    #clfPlot = svm.SVC(kernel='rbf', gamma=0.1, C=C, decision_function_shape='ovr')#for plotting purposes
    clfPlot = clf
    clfPlot.fit(Xtrain,yTrain)

    #predict(Xtest, clf, yTest) #uncomment this to predict the input
    #return clf, clfPlot #Uncomment this to be able to plot the classifier

def evaluateFeatures(X, y):
    forest = ExtraTreesClassifier(n_estimators = 250, random_state = 0)
    forest.fit(X,y)

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


def predict(Xtest, clf, yTest):
    global dataPoints
    predictions = clf.predict(Xtest)
    #clf = joblib.load('testMachineState.pkl')#loads the machine-learning state
    print(accuracy_score(yTest, predictions))


def saveMachinestate(clf):
    joblib.dump(clf, 'testMachineState.pkl')


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
