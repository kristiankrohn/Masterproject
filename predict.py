import time
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import features
import dataset
from globalconst import  *
import globalvar as glb
import classifier
yTestGUI = []
predictionsGUI = []
import matplotlib.pyplot as plt



def predictGUI(clf, y, windowLength, scaler):
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
                X, 0, featuremask=[0,1,2,3,4,5,6,7,9,10,12,13,15,17,18,19,20,21,22,23,25,26], printTime=False)
    Xtest = classifier.realTimeScale(Xtest)
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



def predictRealTime(clf, scaler, debug=False):
    #print(X)
    start = time.time()
    X = dataset.shapeArray(shortLength, checkTimestamp=False)
    #print(len(X[0][0]))
    if (X == -1) or ((len(X[0][0]) < 250) and (len(glb.data[0][filterdata]) > 1500)):
        print("Error from shape array")
    else:
        #Xtest = features.extractFeatures(X, 0)
        #L = np.arange(0, len(X[0][0])/glb.fs, 1/glb.fs)
   
        Xtest = features.extractFeaturesWithMask(
                X, 0, featuremask=[0,1,2,3,4,5,6,7,9,10,12,13,15,17,18,19,20,21,22,23,25,26], printTime=False)
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
        #print("The prediction is: %d" %prediction[0])
        with glb.predictionslock:
            glb.predictions.append(prediction[0])

        if debug:
            timeStop = time.time()
            
            #print()
            print("Time taken to predict with given examples:")
            print(timeStop - start)


