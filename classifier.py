import time
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from globalconst import  *
import globalvar
import itertools




def predict(Xtest, clf, yTest):
    print("Starting to predict")
    start = time.time()
    predictions = clf.predict(Xtest)
    stop = time.time()
    print("Time taken to predict with given examples:")
    print(stop-start)

    #Print the test data to see how well it performs.
    confusionMatrix = confusion_matrix(yTest, predictions, labels = [0,5,2,4,6,8])
    plotConfusionMatrix(confusionMatrix, ["blink","straight", "down", "left", "right", "up"])

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
    return accuracyScore, precision, classificationReport, f1Score


def predictTimer(Xtest, clf, yTest):
    predictionTime = []
    print("Starting to predict")
    for i in range(10000):
        start = time.time()
        predictions = clf.predict(Xtest)
        stop = time.time()
        predictionTime.append(stop-start)
    print("Time taken to predict with given examples:")
    print(min(predictionTime))

    confusionMatrix = confusion_matrix(yTest, predictions, labels = [0,5,2,4,6,8])
    plotConfusionMatrix(confusionMatrix, ["blink","straight", "down", "left", "right", "up"])

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
    return accuracyScore, precision, classificationReport, f1Score

def plotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def tuneSvmParameters(XLtrain, yTrain, XLtest, yTest, debug=True, fast=False, n_jobs=1):
    bestParams = []
    tunedParameters = [{'kernel': ['rbf'], 'gamma': [1e0, 1e-1, 1e-2, 1e-3, 1e-4],
                        'C': [1, 10, 50, 100, 500, 1000, 10000]},
                        {'kernel': ['linear'], 'C': [1, 10, 50, 100, 500, 1000, 10000]}]

    scores = ['precision', 'recall']

    #for score in scores:

    #CV = ?????Increase CV = less variation, more computation time
    #comment in for loop and add score where scores[0] is atm, to make best parameters for prediction and recall
    if debug:
        print("Starting to tune parameters")
        print()

    if fast:
        #clf = svm.LinearSVC(penalty = 'l2',  loss='squared_hinge', dual = False, C = 10, random_state = 42)
        clf = svm.SVC(kernel = 'linear', C = 10, decision_function_shape = 'ovr')
        clf.fit(XLtrain, yTrain)
        bestParams.append({'kernel': 'linear', 'C': 10})
    else:
        clf = GridSearchCV(svm.SVC(), tunedParameters, cv=10, scoring='%s_macro' % scores[0], n_jobs=n_jobs)
        clf.fit(XLtrain, yTrain)
        bestParams.append(clf.best_params_)

    if debug:
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
    else:
        yPred = clf.predict(XLtest)
        labels = unique_labels(yTest, yPred)
        report = classification_report(yTest, yPred)
        p, r, f1, s = precision_recall_fscore_support(yTest, yPred,
                                                      labels=labels,
                                                      average=None)

        return bestParams[0], p, r, f1, s, report

def tuneDecisionTreeParameters(XLtrain, yTrain, XLtest, yTest, n_jobs = 1):
    bestParams = []

    sampleLeafPipeline = [{
           "n_estimators" : [9, 18, 27, 36, 45, 54, 63],
           "max_depth" : [1, 5, 10, 15, 20, 25, 30],
           "min_samples_leaf" : [1, 2, 4, 6, 8, 10]}]


    neighborsPipeline = [{"n_neighbors" : [1, 5, 10, 25, 50]}]

    print("Starting to tune parameters")
    print()

    clf = GridSearchCV(neighbors.KNeighborsClassifier(), neighborsPipeline, cv=10, scoring='%s_macro' % 'precision')

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

def makeScaler(XL):
    scaler = StandardScaler()
    scaler.fit(np.array(XL))
    return scaler

def split(XL, y):
    XLtrain, XLtest, yTrain, yTest = train_test_split(XLscaled,
        labels, test_size = 0.2, random_state = 42, stratify = labels)
    return XLtrain, XLtest, yTrain, yTest

def scaleAndSplit(XL, labels, scaler):
    XLscaled = scaler.transform(np.array(XL))
    XLtrain, XLtest, yTrain, yTest = train_test_split(XLscaled,
        labels, test_size = 0.2, random_state = 42, stratify = labels)

    return XLtrain, XLtest, yTrain, yTest, XLscaled

def realTimeScale(XL, scaler):
    XLscaled = scaler.transform(np.array(XL))
    return XLscaled

def saveScaler(scaler, string):
    joblib.dump(scaler, "Scalers"+slash + string + ".pkl")

def loadScaler(string):
    scaler = joblib.load("Scalers"+slash + string + ".pkl")
    return scaler

def saveMachinestate(clf, string):
    joblib.dump(clf, "Classifiers"+slash + string + ".pkl")

def loadMachineState(string):

    clf = joblib.load("Classifiers"+slash + string + ".pkl")
    return clf
