import pyeeg
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#import scikit-learn as sklearn

#fid =


dataPointsTrain = [[]]
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
    for i in range(90):
        index = "{:03}".format(i + 1)
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
    for i in range(90):
        index = "{:03}".format(i + 1)
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
    #print(labels)
    clf, clfPlot = createAndTrain()
    saveMachinestate(clf)
    plotClassifier(clfPlot)



def createDataset():
    global dfaOpen, dfaClosed, pfdOpen, pfdClosed, labels, dataPointsTrain
    featureVector = []

    for i in range(len(dfaOpen)):
        featureVector = [dfaOpen[i], stdOpen[i]]
        dataPointsTrain.append(featureVector)
        labels.append(0)
        featureVector = [dfaClosed[i], stdClosed[i]]
        dataPointsTrain.append(featureVector)
        labels.append(1)

    dataPointsTrain.pop(0) #remove empty list at start position

def createAndTrain():
    global dataPointsTrain, labels

    Xtrain = np.array(dataPointsTrain)


    Xscaled = preprocessing.scale(Xtrain)
    #min_max_scaler = preprocessing.MinMaxScaler()
    #X_train_minmax = min_max_scaler.fit_transform(Xtrain)
    #print(Xscaled)
    Y = np.array(labels)

    C = 100 #SVM regulation parameter
    clf = svm.SVC(kernel='rbf', gamma=0.1, C=C, decision_function_shape='ovr')
    clf.fit(Xscaled,Y)#skaler???
    clfPlot = svm.SVC(kernel='rbf', gamma=0.1, C=C, decision_function_shape='ovr')#for plotting purposes
    clfPlot.fit(Xtrain,Y)

    return clf, clfPlot


def saveMachinestate(clf):
    joblib.dump(clf, 'testMachineState.pkl')

def plotClassifier(clf):
    global dataPointsTrain
    title = 'SVC with RBF-kernel'

    Xtrain = np.array(dataPointsTrain)

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
