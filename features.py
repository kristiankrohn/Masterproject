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
from itertools import compress
from datetime import datetime
from numpy.fft import fft
from numpy import zeros, floor
from sklearn import svm
from sklearn import neighbors
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import math


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

def cov12(X):
	cov = np.cov(X[0], X[1])
	return cov[0][1]

def cov12a(X):
	cov = np.cov(X[0], X[1])
	return cov[1][0]

def cov13(X):
	cov = np.cov(X[0], X[2])
	return cov[0][1]

def cov13a(X):
	cov = np.cov(X[0], X[2])
	return cov[1][0]

def cov14(X):
	cov = np.cov(X[0], X[3])
	return cov[0][1]

def cov14a(X):
	cov = np.cov(X[0], X[3])
	return cov[1][0]

def cov34(X):
	cov = np.cov(X[2], X[3])
	return cov[0][1]

def cov34a(X):
	cov = np.cov(X[2], X[3])
	return cov[1][0]

def ptp1(X):
	return np.ptp(X[0])

def ptp4(X):
	return np.ptp(X[3])

def pfd1(X):
	return pyeeg.pfd(X[0])

def pfd4(X):
	return pyeeg.pfd(X[3])

def hfd1(X):
	return pyeeg.hfd(X[0], 50) #Okende tall gir viktigere feature, men mye lenger computation time

def hfd4(X):
	return pyeeg.hfd(X[3], 50) #Okende tall gir viktigere feature, men mye lenger computation time

def max1(X):
	return np.amax(X[0])

def min1(X):
	return np.amin(X[0])

def minDiff(X):
	return np.amin(X[1]) - np.amin(X[3])

def maxDiff(X):
	return np.amax(X[1]) - np.amax(X[3])

def specEntropy1(X):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[0], frequencyBands, glb.fs)
	return pyeeg.spectral_entropy(X[0], [0.1, 4, 7, 12,30], 250, powerRatio)

def specEntropy4(X):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[3], frequencyBands, glb.fs)
	return pyeeg.spectral_entropy(X[3], [0.1, 4, 7, 12,30], 250, powerRatio)

def power1a(X):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[0], frequencyBands, glb.fs)
	return power[0]

def power1b(X):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[0], frequencyBands, glb.fs)
	return power[1]

def power1c(X):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[0], frequencyBands, glb.fs)
	return power[2]

def power4a(X):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[3], frequencyBands, glb.fs)
	return power[0]

def power4b(X):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[3], frequencyBands, glb.fs)
	return power[1]

def power4c(X):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[3], frequencyBands, glb.fs)
	return power[2]

def pearsonCoeff14(X):
	pearsonCoefficients14 = np.corrcoef(X[0], X[3])
	return pearsonCoefficients14[1][0]

def pearsonCoeff14a(X):
	pearsonCoefficients14 = np.corrcoef(X[0], X[3])
	return pearsonCoefficients14[0][1]

def pearsonCoeff13(X):
	pearsonCoefficients13 = np.corrcoef(X[0], X[2])
	return pearsonCoefficients13[1][0]

def pearsonCoeff13a(X):
	pearsonCoefficients13 = np.corrcoef(X[0], X[2])
	return pearsonCoefficients13[0][1]

def pearsonCoeff12(X):
	pearsonCoefficients12 = np.corrcoef(X[0], X[1])
	return pearsonCoefficients12[1][0]

def pearsonCoeff12a(X):
	pearsonCoefficients12 = np.corrcoef(X[0], X[1])
	return pearsonCoefficients12[0][1]

def pearsonCoeff34(X):
	pearsonCoefficients12 = np.corrcoef(X[2], X[3])
	return pearsonCoefficients12[1][0]

def pearsonCoeff34a(X):
	pearsonCoefficients12 = np.corrcoef(X[2], X[3])
	return pearsonCoefficients12[0][1]

def stdDeviation(X):
	return np.std(X[0])

def stdDeviation4(X):
	return np.std(X[3])

def slope(X):
	maxIndex = np.argmax(X[0])
	minIndex = np.argmin(X[0])
	minValueCh1 = np.amin(X[0])
	maxValueCh1 = np.amax(X[0])
	slopeCh1 = (minValueCh1 - maxValueCh1)/ (minIndex - maxIndex)
	return slopeCh1

def slope4(X):
	maxIndex = np.argmax(X[3])
	minIndex = np.argmin(X[3])
	minValueCh4 = np.amin(X[3])
	maxValueCh4 = np.amax(X[3])
	slopeCh4 = (minValueCh4 - maxValueCh4)/ (minIndex - maxIndex)
	return slopeCh4

def thetaBeta1(X):
	frequencyBands = [0.1, 4, 8, 12,30]
	bandAvgAmplitudesCh1 = getBandAmplitudes(X[0], frequencyBands)
	thetaBetaRatioCh1 = bandAvgAmplitudesCh1[1]/bandAvgAmplitudesCh1[3]
	return thetaBetaRatioCh1

def thetaBeta4(X):
	frequencyBands = [0.1, 4, 8, 12,30]
	bandAvgAmplitudesCh4 = getBandAmplitudes(X[3], frequencyBands)
	thetaBetaRatioCh4 = bandAvgAmplitudesCh4[1]/bandAvgAmplitudesCh4[3]
	return thetaBetaRatioCh4

def extrema(X):
	extremaFeature = None
	if (np.argmax(X[2]) - np.argmax(X[3])) > 15:
		extremaFeature = 1
	elif(np.argmax(X[2]) - np.argmax(X[3])) < -15:
		extremaFeature = -1
	else:
		extremaFeature = 0
	return extremaFeature

'''
FUNC_MAP = {0: hfd1,
			1: minDiff,
			2: maxDiff,
			3: specEntropy1,
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
			14: pfd1,
			15: ptp1,
			16: slope4,
			17: cov12,
			18: cov12a,
			19: cov14,
			20: cov14a,
			21: cov34,
			22: cov34a,
			23: power1c,
			24: power4c,
			25: stdDeviation4,
			26: ptp4,
			27: min1,
			28: pearsonCoeff12,
			29: pearsonCoeff12a}
'''

FUNC_MAP = {0: hfd1,
			1: hfd4,
			2: minDiff,
			3: maxDiff,
			4: specEntropy1,
			5: specEntropy4,
			6: pearsonCoeff34,
			7: pearsonCoeff34a,
			8: pearsonCoeff14,
			9: pearsonCoeff14a,
			10: cov34,
			11: cov34a,
			12: cov14,
			13: cov14a,
			14: stdDeviation,
			15: stdDeviation4,
			16: slope,
			17: slope4,
			18: thetaBeta1,
			19: thetaBeta4,
			#20: power1a,
			#21: power1b,
			#22: power1c,
			#20: power4a,
			#24: power4b,
			#25: power4c,
			20: pfd1,
			21: pfd4,
			22: ptp1,
			23: ptp4,
			24: min1,
			25: max1
			}

def extractFeaturesWithMask(x,featuremask, printTime=False):
	XL = [[]]
	X = list(x)
	for i in range(len(X[0])):
		startTime = time.time()
		featureVector = []
		Xi = [X[k][i] for k in range(len(X))]

		for j in featuremask:
			 feature = FUNC_MAP[j](Xi)
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

	elif isinstance(p, list):
		print("Featuremask is a list")
		for m in range(len(p)):
			functionstring = str(FUNC_MAP.get(p[m]))
			functionname = functionstring.split(" ")
			returnString += functionname[1] + " "
		return returnString
	else:
		#p = tuple(map(int, p[1:-1].split(',')))
		print(type(p))
		return str(p)

def compareFeatures2(name, shift, windowLength, n_jobs=-1, X = None, y = None, plot=True):
	#datasetfile = "longdata.txt"
	datasetfile = "data.txt"
	merge = True

	if (X == None) or (y == None):
		X, y = dataset.loadDataset(filename=datasetfile, filterCondition=True,
	                                filterType="DcNotch", removePadding=True, shift=shift, windowLength=windowLength)
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
	XL = extractFeaturesWithMask(X, featuremask=range(len(FUNC_MAP)))
	scaler = classifier.makeScaler(XL)
	XLtrain, XLtest, yTrain, yTest, XL = classifier.scaleAndSplit(XL, y[0], scaler)
	#scaler = StandardScaler()
	#XL = scaler.fit_transform(XL, y[0])
	#XLtest = scaler.fit_transform(XLtest, yTest)


	clf = svm.SVC(kernel="linear", C = 10, decision_function_shape = 'ovr')
	#clf = svm.LinearSVC(penalty = 'l2',  loss='squared_hinge', dual = False, C = 10, random_state = 42)
	#clf = RandomForestClassifier(n_estimators = 45, max_depth = 10,  min_samples_leaf = 1, random_state = 40)
	#clf = svm.LinearSVC(penalty = 'l2', dual = False, C = 10, random_state = 42)
	#clf = linear_model.SGDClassifier(penalty = 'l2', random_state = 42)


	rfecv = RFECV(estimator=clf, step=1, cv=10, n_jobs=n_jobs,
	              scoring='accuracy')
	rfecv.fit(XL, y[0])

	print("Optimal number of features : %d" % rfecv.n_features_)
	print("Optimal features: ")
	print(rfecv.support_)

	writeFeatureMask(rfecv.support_, name)

	print("The ranking of the features: ")
	print(rfecv.ranking_)
	print("The scores for each feature combination:")
	print(rfecv.grid_scores_)
	if plot:
		# Plot number of features VS. cross-validation scores
		plt.figure()
		plt.xlabel("Number of features selected")
		plt.ylabel("Cross validation score (nb of correct classifications)")
		plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
		plt.show()

	print("After feature selection: ")
	scores = cross_val_score(rfecv.estimator_, XLtrain, yTrain, cv=10, scoring = 'accuracy')
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	print()
	print("Scores")
	print(scores)

def writeFeatureMask(mask, name):
	
	if isinstance(mask[0], (np.ndarray, np.generic) ):
		print("Converting boolean featuremask to numeric")
		features = range(len(FUNC_MAP))
		featuremaskList = list(compress(features, mask))
	else:
		featuremaskList = mask
	featuremask = open("Featuremask"+slash+name+".txt", 'w+')
	featuremask.write(str(featuremaskList))
	featuremask.close()

def readFeatureMask(name):
	featuremaskFile = open("Featuremask"+slash+name+".txt", 'r')
	featuremaskString = featuremaskFile.read()
	featuremaskFile.close()
	featuremaskString = featuremaskString[1:-1]

	featuremaskList = map(int, featuremaskString.split(', '))
	print featuremaskList
	return featuremaskList



def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def compareFeatures(n_jobs=1, datasetnum=1, shift=False, windowLength=250, zeroClassMultiplier=1):
	#array declaration
	trainings = 0
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
	skip = False
	logging = False



	print("Setup for brute force testing of all feature combination, features in list: %d" %len(FUNC_MAP))
	print("Enter maximum number of features: (\"all\" for all combinations)")
	inputString = raw_input()
	if inputString.isdigit():
		inputval = int(inputString)
		if inputval > len(FUNC_MAP):
			print("Invalid input, exiting")
			return
		if inputval >= 1:
			maxNumFeatures = inputval
		else:
			print("Invalid input, exiting")
			return
	else:
		if inputString == "all":
			maxNumFeatures = len(FUNC_MAP)
			minNumFeatures = 1
			skip = True
		else:
			print("Invalid input, exiting")
			return

	if not skip:
		print("Enter minimum number of features: ")
		inputString = raw_input()
		if inputString.isdigit():
			inputval = int(inputString)
			if inputval <= 0:
				print("Invalid input, exiting")
				return
		else:
			print("Invalid input, exiting")
			return
		if inputval >= 1:
			minNumFeatures = inputval
		else:
			print("Invalid input, exiting")
			return
	print("Is this a debug session?(Will leak memory) [Y/n]")
	inputString = raw_input()
	if inputString == "Y":
		debug = True
		print("Debug session activated, results will not be valid and memory will explode.")
		sendMail = False
		logging = False
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

		print("Do you want to save logs? [Y/n]")
		inputString = raw_input()
		if inputString == "Y":
			logging = True
			print("Saving logs")
		else:
			print("Do not save logs")
			logging = False
	#Load dataset
	#print("Before load")
	'''
	X, y = dataset.loadDataset(filename="data.txt", filterCondition=True,
                                filterType="DcNotch", removePadding=True, shift=False, windowLength=200)
	#print("After load")
	#print X
	if datasetfile == "longdata.txt":
		classes = [0,5,6,4,2,8]
	else:
		classes = [0,1,2,3,4,5,6,7,8,9]
		#classes = [9,7,3,1,0,5]

	X, y = dataset.sortDataset(X, y, length=1000, classes=classes, merge=merge, zeroClassMultiplier=1.2)
	if merge:
		classes = [0,5,6,4,2,8]
			#y = dataset.mergeLabels(y)
	else:
		classes = [0,1,2,3,4,5,6,7,8,9]
	#Calculate features
	#XL = extractAllFeatures(X, channel=0)
	XL = extractFeaturesWithMask(X, featuremask=range(len(FUNC_MAP)))
	#XLtrain, XLtest, yTrain, yTest = classifier.scaleAndSplit(XL, y[0])
	scaler = classifier.makeScaler(XL)
	XLtrain, XLtest, yTrain, yTest, XL = classifier.scaleAndSplit(XL, y[0], scaler)
	'''

	XL = [[],[],[],[],[],[],[],[]]
	y = [[],[],[],[],[],[],[],[]]
	XLlist = []
	ylist = []
	XLtrain = None
	XLtest = None
	yTrain = None
	yTest = None


    ##### Code

	if isinstance(datasetnum, int):
		var = datasetnum
		datasetnum = []
		datasetnum.append(var)

	print(datasetnum)

	for i in datasetnum:
		print i
		dataset.setDatasetFolder(i)

		X, Y = dataset.loadDataset(filename=datasetfile, filterCondition=True,
                                    filterType="DcNotch", removePadding=True, 
                                    shift=shift, windowLength=windowLength)

		Xl, Y = dataset.sortDataset(X, Y, length=130, classes=[0,1,2,3,4,5,6,7,8,9], 
                                        merge = True, zeroClassMultiplier=zeroClassMultiplier)
        
		XLlist.append(Xl)
		ylist.append(Y)
		XL, y = dataset.mergeDatasets(XL, Xl, y, Y)


	for i in range(len(XLlist)):
		XLlist[i] = extractFeaturesWithMask(
                XLlist[i], featuremask=range(len(FUNC_MAP)), printTime=False)
	print("XL list featureextraction finished")
    
	XL = extractFeaturesWithMask(
                XL, featuremask=range(len(FUNC_MAP)), printTime=False)
	print("XL featureextraction finished")
    
	scaler = classifier.makeScaler(XL)

	for i in range(len(XLlist)):
		XLtrain1, XLtest1, yTrain1, yTest1, XLlist[i] = classifier.scaleAndSplit(XLlist[i], ylist[i][0], scaler)
		if i == 0:
			XLtrain = XLtrain1
			yTrain = yTrain1
			XLtest = XLtest1
			yTest = yTest1
		else:
			XLtrain, yTrain = dataset.mergeDatasets(XLtrain, XLtrain1, yTrain, yTrain1)
			XLtest, yTest = dataset.mergeDatasets(XLtest, XLtest1, yTest, yTest1)
	print("Split fininshed, starting training")
    


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
		#numberOfCombinations += len(list(combinations(features, i))) #Dette sprenger minnet :/
		comb = nCr(len(features),i)
		print("Number of iterations for combinations of length %d: %d" %(i, comb))
		numberOfCombinations += comb

	print("Number of combinations to test %d" %numberOfCombinations)

	#for i in range(minNumFeatures, maxNumFeatures+1):
	for i in range(maxNumFeatures, minNumFeatures-1, -1):

		print("Starting to read PermutationLog")
		try:
			permfile = open("Logs"+slash+"PermutationLog"+ str(i)+".txt", 'r')
			#permfile = open("Logs"+slash+"PermutationLog.txt", 'r')

		except IOError:
			print("PermutationLog file does not exist")
			skip = False
		else:
			print("Performing operations on PermutationLog buffer")
			PermutationsString = permfile.read()
			permfile.close()
			PermutationsList = PermutationsString.split(':')
			PermutationsList.pop(0)
			#PermutationsList = tuple(PermutationsList)
			#Might need some more processing, now returns a list of tuples
			#print PermutationsList[28]

			for q in range(len(PermutationsList)):
				#print(eval(PermutationsList[i]))
				#PermutationsList[i] = tuple(map(int, PermutationsList[i][1:-1].split(',')))
				PermutationsList[q] = tuple(eval(PermutationsList[q]))
			print("Finished with operations")
			skip = True
		start = datetime.now()
		print("Finished reading permutations file")
		lastTrainings = 1000
		for p in combinations(features, i): #If order matters use permutations

			if skip == True:

				if p in PermutationsList:
					#print("Combination exists")
					numberOfCombinations -= 1
					trainings += 1
					if trainings == lastTrainings:
						lastTrainings = trainings + 1000
						print("Training number: %d" %trainings)
						print("Remaining combinations: %d" %numberOfCombinations)
						print("Elapsed time for checking that this combination exists: " + str(elapsedTime))

					stop = datetime.now()
					elapsedTime = (stop-start)
					start = stop
				else:
					print("Found Starting point")
					skip = False


			if skip == False:

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
				#print("Starting to train with combination: "+convertPermutationToFeatureString(p))


				bestParams, presc, r, f1, s, report = classifier.tuneSvmParameters(XLtrainPerm,
															yTrain, XLtestPerm, yTest,
															debug=False, fast=True,
															n_jobs = n_jobs)




				if logging:

					permfile = open(dir_path+slash+"Logs"+slash+"PermutationLog"+ str(i)+".txt", 'a+')
					permfile.write(":")
					permfile.write(str(p))
					permfile.close()


					permfile = open(dir_path+slash+"Logs"+slash+"PrecisionLog"+ str(i)+".txt", 'a+')
					permfile.write(":")
					for k in range(len(presc)):
						permfile.write(','+str(presc[k]))
					permfile.close()

					permfile = open(dir_path+slash+"Logs"+slash+"RecallLog"+ str(i)+".txt", 'a+')
					permfile.write(":")
					for k in range(len(r)):
						permfile.write(','+str(r[k]))
					permfile.close()

					permfile = open(dir_path+slash+"Logs"+slash+"F1Log"+ str(i)+".txt", 'a+')
					permfile.write(":")
					for k in range(len(f1)):
						permfile.write(','+str(f1[k]))
					permfile.close()

				if debug:
					#Append scores
					allPermutations.append(p)
					allParams.append(bestParams)
					allP.append(presc)
					allPavg.append(np.average(presc, weights=s))
					allR.append(r)
					allF1.append(f1)

					winner = allPavg.index(max(allPavg)) #Check for max average precision
					print(report)
					print("Best features so far are: " + convertPermutationToFeatureString(allPermutations[winner]))
					print("Best result so far are: ", allPavg[winner])

				#print("Best parameters for this feature combination: " + str(bestParams))
				stop = datetime.now()
				numberOfCombinations -= 1
				trainings += 1
				remainingTime = (stop-start)*numberOfCombinations
				elapsedTime = (stop-start)
				print("Training number: %d" %trainings)
				print("Remaining combinations: %d" %numberOfCombinations)
				print("Elapsed time for training with this combination: " + str(elapsedTime))
				print("Estimated remaining time: " + str(remainingTime))
	'''
	#Evaluate score
	if len(allPavg) > 1:
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
	'''
	if sendMail:
	    mail.sendemail(from_addr    = 'dronemasterprosjekt@gmail.com',
						to_addr_list = ['krishk@stud.ntnu.no','adriari@stud.ntnu.no'],
						cc_addr_list = [],
						subject      = "Training finished with combinations of %d to %d features" %(minNumFeatures, maxNumFeatures),
						message      = "Logs are ready for download ",
						login        = 'dronemasterprosjekt',
						password     = 'drone123')



def main():
	#cleanLogs()
	#compareFeatures2(n_jobs=-1)
	compareFeatures(-1, datasetnum = [1,2], shift=False, windowLength=250, zeroClassMultiplier=2)
	#mask = readFeatureMask("BruteForcelowenergyaverageprecision9")
	#print(convertPermutationToFeatureString(mask))
if __name__ == '__main__':
	main()
