import numpy as np
import pyeeg
import time
import dataset
from globalconst import  *
import globalvar as glb
import copy
import mail
import classifier
from itertools import permutations
from itertools import combinations
from datetime import datetime
from numpy.fft import fft
from numpy import zeros, floor
from sklearn import svm
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

def getBandAmplitudes(X, Band):
	Fs = glb.fs
	C = fft(X)
	C = abs(C)
	avgAmplitude =zeros(len(Band)-1);

	for Freq_Index in range(len(Band)-1):
		Freq = float(Band[Freq_Index])										## Xin Liu
		Next_Freq = float(Band[Freq_Index+1])
		#Endret til Int for aa faa det til aa gaa igjennom
		avgAmplitude[Freq_Index] = sum(C[int(Freq/Fs*len(X)):int(Next_Freq/Fs*len(X))]) / len(C[int(Freq/Fs*len(X)):int(Next_Freq/Fs*len(X))])
	return avgAmplitude


def extractFeatures(X, channel):
    XL = [[]]
    frequencyBands = [0.1, 4, 8, 12,30]
    Fs = 250
    featureVector = []

    for i in range(len(X[0])):
        startTime = time.time()
        power, powerRatio = pyeeg.bin_power(X[channel][i], frequencyBands, Fs)
        bandAvgAmplitudesCh1 = getBandAmplitudes(X[0][i], frequencyBands)
        bandAvgAmplitudesCh2 = getBandAmplitudes(X[1][i], frequencyBands)
        bandAvgAmplitudesCh3 = getBandAmplitudes(X[0][i], frequencyBands)
        bandAvgAmplitudesCh4 = getBandAmplitudes(X[1][i], frequencyBands)
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

        maxIndex = np.argmax(list(X[0][i]))
        minIndex = np.argmin(list(X[0][i]))
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
                        #np.argmax(list(X[2][i])) - np.argmax(list(X[3][i])), #This seems promising, needs more testing. Index for max point
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



def cov12(X, channel):
	cov = np.cov(X[0], X[1])
	return cov[0][1]
def cov12a(X, channel):
	cov = np.cov(X[0], X[1])
	return cov[1][0]
def cov14(X, channel):
	cov = np.cov(X[0], X[3])
	return cov[0][1]

def cov14a(X, channel):
	cov = np.cov(X[0], X[3])
	return cov[1][0]

def cov34(X, channel):
	cov = np.cov(X[2], X[3])
	return cov[0][1]
def cov34a(X, channel):
	cov = np.cov(X[2], X[3])
	return cov[1][0]

def ptp1(X, channel):
	return np.ptp(X[0])
def ptp4(X, channel):
	return np.ptp(X[3])

def pfd(X, channel):
	return pyeeg.pfd(X[channel])

def hfd(X, channel):
	return pyeeg.hfd(X[channel], 50) #Okende tall gir viktigere feature, men mye lenger computation time

def max1(X, channel):
	return np.amax(X[0])

def minDiff(X, channel):
	return np.amin(X[0]) - np.amin(X[2])

def maxDiff(X, channel):
	return np.amax(X[0]) - np.amax(X[2])

def specEntropy(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[channel], frequencyBands, glb.fs)
	return pyeeg.spectral_entropy(X[channel], [0.1, 4, 7, 12,30], 250, powerRatio)
def power1(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[0], frequencyBands, glb.fs)
	return power[2]
def power4(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[3], frequencyBands, glb.fs)
	return power[2]

def pearsonCoeff14(X, channel):
	pearsonCoefficients14 = np.corrcoef(X[0], X[3])
	pearsonCoefficients13 = np.corrcoef(X[0], X[2])
	return pearsonCoefficients14[1][0]
def pearsonCoeff14a(X, channel):
	pearsonCoefficients14 = np.corrcoef(X[0], X[3])
	pearsonCoefficients13 = np.corrcoef(X[0], X[2])
	return pearsonCoefficients14[0][1]

def pearsonCoeff13(X, channel):
	pearsonCoefficients13 = np.corrcoef(X[0], X[2])
	return pearsonCoefficients13[1][0]
def pearsonCoeff13a(X, channel):
	pearsonCoefficients13 = np.corrcoef(X[0], X[2])
	return pearsonCoefficients13[0][1]

def stdDeviation(X, channel):
	return np.std(X[channel])

def stdDeviation4(X, channel):
	return np.std(X[3])

def slope(X, channel):
	maxIndex = np.argmax(X[channel])
	minIndex = np.argmin(X[channel])
	minValueCh1 = np.amin(X[0])
	maxValueCh1 = np.amax(X[0])
	slopeCh1 = (minValueCh1 - maxValueCh1)/ (minIndex - maxIndex)
	return slopeCh1
def slope4(X, channel):
	channel = 3
	maxIndex = np.argmax(X[channel])
	minIndex = np.argmin(X[channel])
	minValueCh4 = np.amin(X[3])
	maxValueCh4 = np.amax(X[3])
	slopeCh4 = (minValueCh4 - maxValueCh4)/ (minIndex - maxIndex)
	return slopeCh4

def thetaBeta1(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	bandAvgAmplitudesCh1 = getBandAmplitudes(X[0], frequencyBands)
	thetaBetaRatioCh1 = bandAvgAmplitudesCh1[1]/bandAvgAmplitudesCh1[3]
	return thetaBetaRatioCh1
def thetaBeta4(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	bandAvgAmplitudesCh4 = getBandAmplitudes(X[3], frequencyBands)
	thetaBetaRatioCh4 = bandAvgAmplitudesCh4[1]/bandAvgAmplitudesCh4[3]
	return thetaBetaRatioCh4

def extrema(X, channel):
	extremaFeature = None
	if (np.argmax(X[2]) - np.argmax(X[3])) > 15:
		extremaFeature = 1
	elif(np.argmax(X[2]) - np.argmax(X[3])) < -15:
		extremaFeature = -1
	else:
		extremaFeature = 0
	return extremaFeature

FUNC_MAP = {0: hfd,
			1: minDiff,
			2: maxDiff,
			3: specEntropy,
			4: pearsonCoeff14,
			5: stdDeviation,
			6: slope,
			7: thetaBeta1,
			8: extrema,
			9: pearsonCoeff13,
			10: pearsonCoeff14a,
			11: pearsonCoeff13a,
			12: thetaBeta4,
			13: max1,
			14: pfd,
			15: ptp1,
			16: slope4,
			17: cov12,
			18: cov12a,
			19: cov14,
			20: cov14a,
			21: cov34,
			22: cov34a,
			23: power1,
			24: power4,
			25: stdDeviation4,
			26: ptp4}

def extractFeaturesWithMask(x, channel, featuremask, printTime=False):
	XL = [[]]
	X = list(x)
	for i in range(len(X[0])):
		startTime = time.time()
		featureVector = []
		Xi = [X[k][i] for k in range(len(X))]

		for j in featuremask:
			 feature = FUNC_MAP[j](Xi, channel)
			 featureVector.append(feature)
		XL.append(featureVector)
		if printTime:
			print("Time taken to extract features for example %d: " % i)
			print(time.time() - startTime)
		#print(XL)

	XL.pop(0)
	#print XL
	return XL

def convertPermutationToFeatureString(p):
	returnString = ""
	if isinstance(p, tuple):
		for m in range(len(p)):
			functionstring = str(FUNC_MAP.get(p[m]))
			functionname = functionstring.split(" ")
			returnString += functionname[1] + " "
		return returnString
	else:
		#p = tuple(map(int, p[1:-1].split(',')))
		print(type(p))
		return str(p)

def compareFeatures2(n_jobs=1):
	#datasetfile = "longdata.txt"
	datasetfile = "data.txt"
	merge = True
	X, y = dataset.loadDataset(filename=datasetfile)
	#print("After load")
    #print X
	if datasetfile == "longdata.txt":
		classes = [0,5,6,4,2,8]
	else:
		classes = [0,1,2,3,4,5,6,7,8,9]
		#classes = [9,7,3,1,0,5]

	X, y = dataset.sortDataset(X, y, length=1000, classes=classes, merge=merge) #,6,4,2,8
	if merge:
		classes = [0,5,6,4,2,8]
			#y = dataset.mergeLabels(y)
	else:
		classes = [0,1,2,3,4,5,6,7,8,9]
	#Calculate features
	#XL = extractAllFeatures(X, channel=0)
	XL = extractFeaturesWithMask(X, channel = 0, featuremask=range(len(FUNC_MAP)))
	XLtrain, XLtest, yTrain, yTest = classifier.scaleAndSplit(XL, y[0])
	scaler = StandardScaler()
	XL = scaler.fit_transform(XL, y[0])
	#XLtest = scaler.fit_transform(XLtest, yTest)


	clf = svm.SVC(kernel="linear", C = 50, decision_function_shape = 'ovr')
	#clf = RandomForestClassifier(n_estimators = 54, max_depth = 5,  min_samples_leaf = 1, random_state = 40)
	#clf = svm.LinearSVC(penalty = 'l2', dual = False, C = 50, random_state = 42)
	#clf = linear_model.SGDClassifier(penalty = 'l2', random_state = 42)

	rfecv = RFECV(estimator=clf, step=1, cv=10, n_jobs=n_jobs,
	              scoring='accuracy')
	rfecv.fit(XL, y[0])

	print("Optimal number of features : %d" % rfecv.n_features_)
	print("Optimal features: ")
	print(rfecv.support_)

	print("The ranking of the features: ")
	print(rfecv.ranking_)
	print("The scores for each feature combination:")
	print(rfecv.grid_scores_)
	# Plot number of features VS. cross-validation scores
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.show()

def compareFeatures(n_jobs=1):
	#array declaration
	allPermutations = []
	allParams = []
	allPavg = []
	allP = []
	allR = []
	allF1 = []
	allS = []

	#Constants and configuration
	maxNumFeatures = 8
	minNumFeatures = 6 #Must be bigger than 1
	#datasetfile = "longdata.txt"
	datasetfile = "data.txt"
	merge = True

	print("Setup for brute force testing of all feature combination")
	print("Enter maximum number of features: ")
	inputString = raw_input()
	if inputString.isdigit():
		inputval = int(inputString)
	else:
		print("Invalid input, exiting")
		return
	if inputval >= 1:
		maxNumFeatures = inputval
	else:
		print("Invalid input, exiting")
		return

	print("Enter minimum number of features: ")
	inputString = raw_input()
	if inputString.isdigit():
		inputval = int(inputString)
	else:
		print("Invalid input, exiting")
		return
	if inputval >= 1:
		minNumFeatures = inputval
	else:
		print("Invalid input, exiting")
		return
	print("Is this a debug session? [Y/n]")
	inputString = raw_input()
	if inputString == "Y":
		debug = True
		print("Debug session activated, results will not be valid.")
		sendMail = False
	else:
		debug = False
		print("Normal session activated, results will be valid. ")
		print("Do you want to send a mail notification when finished? [Y/n]")
		inputString = raw_input()
		if inputString == "Y":
			sendMail = True
			print("Sending mail when script is finished")
		else:
			print("Do not send mail when script is finished")
			sendMail = False

	#Load dataset
	#print("Before load")
	X, y = dataset.loadDataset(filename=datasetfile)
	#print("After load")
    #print X
	if datasetfile == "longdata.txt":
		classes = [0,5,6,4,2,8]
	else:
		classes = [0,1,2,3,4,5,6,7,8,9]
		#classes = [9,7,3,1,0,5]

	X, y = dataset.sortDataset(X, y, length=1000, classes=classes, merge=merge) #,6,4,2,8
	if merge:
		classes = [0,5,6,4,2,8]
			#y = dataset.mergeLabels(y)
	else:
		classes = [0,1,2,3,4,5,6,7,8,9]
	#Calculate features
	#XL = extractAllFeatures(X, channel=0)
	XL = extractFeaturesWithMask(X, channel = 0, featuremask=range(len(FUNC_MAP)))
	XLtrain, XLtest, yTrain, yTest = classifier.scaleAndSplit(XL, y[0])

	features = range(len(XL[0]))
	print("Featureextraction finished, number of features to check: %d"%len(XL[0]))

	if len(features) < maxNumFeatures:
		maxNumFeatures = len(features)
	elif maxNumFeatures < minNumFeatures:
		maxNumFeatures = minNumFeatures

	if minNumFeatures < 1:
		minNumFeatures = 1
	elif minNumFeatures > maxNumFeatures:
		minNumFeatures = maxNumFeatures
	print("Testing with combinations of %d to %d" %(minNumFeatures, maxNumFeatures))
	numberOfCombinations = 0
	for i in range(minNumFeatures, maxNumFeatures+1):
		numberOfCombinations += len(list(combinations(features, i)))
	print("Number of combinations to test %d" %numberOfCombinations)
	for i in range(minNumFeatures, maxNumFeatures+1):
	    #print list(combinations(features, i))

		for p in combinations(features, i): #If order matters use permutations
            #print p
			start = datetime.now()
			XLtrainPerm = np.empty([len(XLtrain), i])
			XLtestPerm = np.empty([len(XLtest), i])
			for j in range(len(XLtrain)):
				#print j
				#print([XLtrain[j][k] for k in p])
				XLtrainPerm[j] = [XLtrain[j][k] for k in p]
			for j in range(len(XLtest)):
				#print j
				#print([XLtest[j][k] for k in p])
				XLtestPerm[j] = [XLtest[j][k] for k in p]
			#print(XLtrainPerm[0])
			#print(XLtestPerm[0])
			print("Starting to train with combination: "+convertPermutationToFeatureString(p))
			#print(p)

			#print(len(XLtrainPerm))
			#Optimize setting

			#def crossValidation(p, XLtrainPerm, yTrain, XLtestPerm, yTest)

			bestParams, presc, r, f1, s, report = classifier.tuneSvmParameters(XLtrainPerm,
			                                            yTrain, XLtestPerm, yTest,
			                                            debug=False, fast=debug,
			                                            n_jobs = n_jobs)

			#Append scores

			logfile = open("Logs"+slash+"Logfile.txt", 'a+')
			logfile.write("Feature combination:")
			logfile.write(str(p)+"\n")
			logfile.write(report+"\n\n\n")
			logfile.close()


			allPermutations.append(p)
			permfile = open("Logs"+slash+"PermutationLog.txt", 'a+')
			permfile.write(":")
			permfile.write(str(p))
			permfile.close()

			allParams.append(bestParams)
			permfile = open("Logs"+slash+"ParameterLog.txt", 'a+')
			permfile.write(";")
			permfile.write(str(bestParams))
			permfile.close()

			allP.append(presc)
			permfile = open("Logs"+slash+"PrecisionLog.txt", 'a+')
			permfile.write(":")
			for k in range(len(presc)):
				permfile.write(','+str(presc[k]))
			permfile.close()
			allPavg.append(np.average(presc, weights=s))

			allR.append(r)
			permfile = open("Logs"+slash+"RecallLog.txt", 'a+')
			permfile.write(":")
			for k in range(len(r)):
				permfile.write(','+str(r[k]))
			permfile.close()

			allF1.append(f1)
			permfile = open("Logs"+slash+"F1Log.txt", 'a+')
			permfile.write(":")
			for k in range(len(f1)):
				permfile.write(','+str(f1[k]))
			permfile.close()

			allS.append(s)
			permfile = open("Logs"+slash+"SupportLog.txt", 'a+')
			permfile.write(":")
			for k in range(len(s)):
				permfile.write(','+str(s[k]))
			permfile.close()

			winner = allPavg.index(max(allPavg)) #Check for max average precision
			print(report)
			print("Best features so far are: " + convertPermutationToFeatureString(allPermutations[winner]))

			print("Pest parameters for this feature combination: " + str(bestParams))
			stop = datetime.now()
			numberOfCombinations -= 1
			remainingTime = (stop-start)*numberOfCombinations
			elapsedTime = (stop-start)
			print("Elapsed time for training with this combination: " + str(elapsedTime))
			print("Estimated remaining time: " + str(remainingTime))
    #Evaluate score

	winner = allPavg.index(max(allPavg)) #Check for max average precision
	p = allPermutations[winner]
	XLtrainPerm = np.empty([len(XLtrain), len(p)])
	XLtestPerm = np.empty([len(XLtest), len(p)])
	p = allPermutations[winner]
	for j in range(len(XLtrain)):
		XLtrainPerm[j] = [XLtrain[j][k] for k in p]
	for j in range(len(XLtest)):
		XLtestPerm[j] = [XLtest[j][k] for k in p]

	print("Best features for max average precision are:")
	print allPermutations[winner]
	#Test
	bestParams = allParams[winner]
	print("Best parameters for max average precision are: ")
	print(bestParams)

	if bestParams['kernel'] == 'linear':
		clf = svm.SVC(kernel =bestParams['kernel'], C = bestParams['C'], decision_function_shape = 'ovr')
	else:
		clf = svm.SVC(kernel = bestParams['kernel'], gamma=bestParams['gamma'], C= bestParams['C'], decision_function_shape='ovr')

	clf.fit(XLtrainPerm, yTrain)
	saveMachinestate(clf, "BruteForceClassifier")
	featuremask = open("featuremask.txt", 'w+')
	featuremask.write(str(allPermutations[winner]))
	#featuremask.write(",")
	featuremask.close()

	yPred = clf.predict(XLtestPerm)
	print(classification_report(yTest, yPred))
	if sendMail:
	    mail.sendemail(from_addr    = 'dronemasterprosjekt@gmail.com',
	                    to_addr_list = ['krishk@stud.ntnu.no','adriari@stud.ntnu.no'],
	                    cc_addr_list = [],
	                    subject      = "Training finished with combinations of %d to %d features" %(minNumFeatures, maxNumFeatures),
	                    message      = "Best result is with these features: "+str(allPermutations[winner]) + "\n"
	                                    + classification_report(yTest, yPred),
	                    login        = 'dronemasterprosjekt',
	                    password     = 'drone123')



def readLogs():
    import ast
    permfile = open("Logs"+slash+"PermutationLog.txt", 'r')
    PermutationsString = permfile.read()
    permfile.close()
    PermutationsList = PermutationsString.split(':')
    PermutationsList.pop(0)
    #PermutationsList = tuple(PermutationsList)
    #Might need some more processing, now returns a list of tuples
    print PermutationsList[28]
    for i in range(len(PermutationsList)):
		#print(eval(PermutationsList[i]))
		#PermutationsList[i] = tuple(map(int, PermutationsList[i][1:-1].split(',')))
		PermutationsList[i] = tuple(eval(PermutationsList[i]))


    permfile = open("Logs"+slash+"ParameterLog.txt", 'r')
    ParametersString = permfile.read()
    permfile.close()
    ParametersList = ParametersString.split(';')
    ParametersList.pop(0)
    ParametersList = [ast.literal_eval(i) for i in ParametersList]

    permfile = open("Logs"+slash+"PrecisionLog.txt", 'r')
    PrecisionString = permfile.read()
    permfile.close()
    PrecisionList = PrecisionString.split(":")
    PrecisionList.pop(0)

    for j in range(len(PrecisionList)):
        PrecisionSubList = PrecisionList[j].split(',')
        PrecisionSubList.pop(0)
        PrecisionList[j] = list([float(i) for i in PrecisionSubList])

    permfile = open("Logs"+slash+"RecallLog.txt", 'r')
    RecallString = permfile.read()
    permfile.close()
    RecallList = RecallString.split(":")
    RecallList.pop(0)

    for j in range(len(RecallList)):
        RecallSubList = RecallList[j].split(',')
        RecallSubList.pop(0)
        RecallList[j] = list([float(i) for i in RecallSubList])

    permfile = open("Logs"+slash+"F1Log.txt", 'r')
    f1String= permfile.read()
    permfile.close()
    f1List = f1String.split(":")
    f1List.pop(0)

    for j in range(len(f1List)):
        f1SubList = f1List[j].split(',')
        f1SubList.pop(0)
        f1List[j] = list([float(i) for i in f1SubList])

    permfile = open("Logs"+slash+"SupportLog.txt", 'r')
    supportString = permfile.read()
    permfile.close()
    supportList = supportString.split(":")
    supportList.pop(0)

    for j in range(len(supportList)):
        supportSubList = supportList[j].split(',')
        supportSubList.pop(0)
        supportList[j] = list([float(i) for i in supportSubList])

    return PermutationsList, ParametersList, PrecisionList, RecallList, f1List, supportList


def evaluateLogs(evaluationParam="maxminprecision"):
    PermutationsList, ParametersList, PrecisionList, RecallList, f1List, supportList = readLogs()
    if evaluationParam == "averageprecision":
        allPavg = []
        for i in range(len(PrecisionList)):
            allPavg.append(np.average(PrecisionList[i], weights=supportList[i]))
        winner = allPavg.index(max(allPavg))
        print("Best features are: " + convertPermutationToFeatureString(PermutationsList[winner]))
        print("Best parameters are: " + str(ParametersList[winner]))
    elif evaluationParam == "maxminprecision":
        allPmin = []
        for i in range(len(PrecisionList)):
            allPmin.append(min(PrecisionList[i]))
        winner = allPmin.index(max(allPmin))
        print("Best features are: " + convertPermutationToFeatureString(PermutationsList[winner]))
        print("Best parameters are: " + str(ParametersList[winner]))
    elif evaluationParam == "maxminrecall":
        allRmin = []
        for i in range(len(RecallList)):
            allRmin.append(min(RecallList[i]))
        winner = allRmin.index(max(allRmin))
        print("Best features are: " + convertPermutationToFeatureString(PermutationsList[winner]))
        print("Best parameters are: " + str(ParametersList[winner]))

    else:
        print("Invalid input")
        return -1
    return PermutationsList[winner], ParametersList[winner]

def cleanLogs():
    logfile = open("Logs"+slash+"Logfile.txt", 'w')
    logfile.truncate(0)
    logfile.close()

    permfile = open("Logs"+slash+"PermutationLog.txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open("Logs"+slash+"ParameterLog.txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open("Logs"+slash+"PrecisionLog.txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open("Logs"+slash+"RecallLog.txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open("Logs"+slash+"F1Log.txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open("Logs"+slash+"SupportLog.txt", 'w')
    permfile.truncate(0)
    permfile.close()


def main():
	compareFeatures2(n_jobs=-1)
	#evaluateLogs("maxminrecall")
if __name__ == '__main__':
	main()
