import time
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import features
import dataset
from globalconst import  *
import globalvar as glb
import classifier
yTestGUI = []
predictionsGUI = []
import matplotlib.pyplot as plt
import dill as pickle

def createPredictor(name, windowLength, shift = None):
    ##### Parameters

    if windowLength < 250 and shift == None:
        shift = True
        print "Shift is true"
    else:
        shift = False
        print "Shift is false"

    ##### Save parameters
    parameters = {'windowLength': windowLength, 'shift': shift}
    pickle.dump(parameters, open( "Parameters" + slash + name + ".pkl", "wb" ) )
    ##### Declarations
    bestParams = []
    accuracyScore = []
    f1Score = []
    precision = []
    classificationReport = []

    ##### Code
    dataset.setDatasetFolder(1)

    X, y = dataset.loadDataset(filename="data.txt", filterCondition=True,
                                filterType="DcNotch", removePadding=True, 
                                shift=shift, windowLength=windowLength)

    X, y = dataset.sortDataset(X, y, length=1000, classes=[0,1,2,3,4,5,6,7,8,9], 
                                    merge = True)

    features.compareFeatures2(name, shift, windowLength, X=X, y=y)
    featuremask = features.readFeatureMask(name)
    
    XL = features.extractFeaturesWithMask(
            X, featuremask=featuremask, printTime=False)

    XLtrain, XLtest, yTrain, yTest, XL, scaler = classifier.scaleAndSplit(XL, y[0])



    clf = svm.SVC(kernel = 'rbf', gamma = 0.01, C = 10, decision_function_shape = 'ovr')
    clf.fit(XLtrain,yTrain)

    classifier.saveMachinestate(clf, name)   #Uncomment this to save the machine state
    classifier.saveScaler(scaler, name)

    scores = cross_val_score(clf, XLtrain, yTrain, cv=50, scoring = 'accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()
    print("Scores")
    print(scores)



    tempAccuracyScore, tempPrecision, tempClassificationReport\
    , tempf1Score = classifier.predict(XLtest, clf, yTest)
    
    accuracyScore.append(tempAccuracyScore)
    f1Score.append(tempf1Score)
    precision.append(tempPrecision)
    classificationReport.append(tempClassificationReport)

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
    print("Classification Report:")
    print(classificationReport[0])



def loadPredictor(name):
    params = pickle.load( open( "Parameters" + slash + name + ".pkl", "rb" ) )
    print params    
    clf = classifier.loadMachineState(name)
    featuremask = features.readFeatureMask(name)
    scaler = classifier.loadScaler(name)
    parameters = {'clf':clf, 'scaler':scaler, 'featuremask':featuremask, 'windowLength':params['windowLength'], 'shift':params['shift']}
    return parameters


def predictGUI(y, clf, scaler, featuremask, windowLength, shift):  ### Trenger testing og implementasjon i gui.py  
    global yTestGUI, predictionsGUI
    MergeDict = {0:0,   1:8,  2:8,  3:6,  4:6,  5:5,  6:4,  7:4,  8:2,  9:2}
    y = MergeDict[y]
    yTest = [y]
    #print("Starting to predict with GUI")
    start = time.time()
    X = dataset.shapeArray(windowLength, y)
    if X == -1:
        return

    Xtest = features.extractFeaturesWithMask(
                X, featuremask=featuremask, printTime=False)
    Xtest = classifier.realTimeScale(Xtest, scaler)
    predictions = clf.predict(Xtest)
    #print("Time taken to predict with given examples in GUI:")
    #print(time.time() - start)
    #Print the test data to see how well it performs.
    if yTest == predictions:
        print("Correct prediction of %d!" %predictions[0])
    else:
        print("Should have predicted: %d Actually predicted: %d" %(y, predictions[0]))
        #print("Actually predicted: %d" %predictions[0])

    yTestGUI.append(y)
    predictionsGUI.append(predictions[0])

    yfile = open("y.txt", 'a+')
    yfile.write(str(y))
    yfile.write(",")
    yfile.close()

    pfile = open("p.txt", 'a+')
    pfile.write(str(predictions[0]))
    pfile.write(",")
    pfile.close()



def resetClassificationReport():
    print("You are about to clear the classificationreport")
    print("Continue? [Y/n]")
    inputString = raw_input()
    if inputString == "Y":
        tempfile = open("y.txt", 'w')
        tempfile.truncate(0)
        tempfile.close()
        tempfile = open("p.txt", 'w')
        tempfile.truncate(0)
        tempfile.close()
        print("Classification cleared successfully")
    else:
        print("Classification report clearing aborted")



def classificationReportGUI():
    #global yTestGUI, predictionsGUI
    yfile = open("y.txt", 'r')
    AllY = yfile.read()
    yfile.close()
    yTestGUI = []
    yTestGUI = AllY.split(",")

    pfile = open("p.txt", 'r')
    AllP = pfile.read()
    pfile.close()
    predictionsGUI = []
    predictionsGUI = AllP.split(",")

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



#def predictRealTime(clf, scaler, featuremask, debug=False):
def predictRealTime(clf, scaler, featuremask, windowLength, shift, debug=False): ### Trenger testing og implementasjon i main
    #print(X)
    start = time.time()
    X = dataset.shapeArray(windowLength, checkTimestamp=False)
    #print(len(X[0][0]))
    if (X == -1) or ((len(X[0][0]) < windowLength) and (len(glb.data[0][filterdata]) > 1500)):
        print("Error from shape array")
    else:
        #Xtest = features.extractFeatures(X, 0)
        #L = np.arange(0, len(X[0][0])/glb.fs, 1/glb.fs)
   
        Xtest = features.extractFeaturesWithMask(
                X, featuremask=featuremask, printTime=False)
        Xtest = classifier.realTimeScale(Xtest, scaler)
        #print Xtest[0]
        if debug:
            L = np.arange(0, len(Xtest[0]))
            plt.ion()
            plt.show()
            plt.clf()
            #for i in range(numCh):
            plt.plot(L, Xtest[0])
            plt.ylabel('uV')
            plt.xlabel('Seconds')
            plt.draw()
            plt.pause(0.001)
            #print(len(Xtest[0]))
            print("Starting to predict")
        
        prediction = clf.predict(Xtest)
        if not(prediction == 5 or prediction == 0):
            #print("The prediction is: %d" %prediction[0])
            pass
        with glb.predictionslock:
            glb.predictions.append(prediction[0])

        if debug:
            timeStop = time.time()
            
            #print()
            print("Time taken to predict with given examples:")
            print(timeStop - start)

def main():
    createPredictor("testing", 100)

if __name__ == '__main__':
    main()

