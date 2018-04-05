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

def cov12(X, channel):
	cov = np.cov(X[0], X[1])
	return cov[0][1]
def cov12a(X, channel):
	cov = np.cov(X[0], X[1])
	return cov[1][0]

def cov13(X, channel):
	cov = np.cov(X[0], X[2])
	return cov[0][1]
def cov13a(X, channel):
	cov = np.cov(X[0], X[2])
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

def pfd1(X, channel):
	return pyeeg.pfd(X[0])

def pfd4(X, channel):
	return pyeeg.pfd(X[3])

def hfd1(X, channel):
	return pyeeg.hfd(X[0], 50) #Okende tall gir viktigere feature, men mye lenger computation time

def hfd4(X, channel):
	return pyeeg.hfd(X[3], 50) #Okende tall gir viktigere feature, men mye lenger computation time

def max1(X, channel):
	return np.amax(X[0])
def min1(X, channel):
	return np.amin(X[0])

def minDiff(X, channel):
	return np.amin(X[1]) - np.amin(X[3])

def maxDiff(X, channel):
	return np.amax(X[1]) - np.amax(X[3])

def specEntropy1(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[0], frequencyBands, glb.fs)
	return pyeeg.spectral_entropy(X[0], [0.1, 4, 7, 12,30], 250, powerRatio)

def specEntropy4(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[3], frequencyBands, glb.fs)
	return pyeeg.spectral_entropy(X[3], [0.1, 4, 7, 12,30], 250, powerRatio)

def power1a(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[0], frequencyBands, glb.fs)
	return power[0]

def power1b(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[0], frequencyBands, glb.fs)
	return power[1]

def power1c(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[0], frequencyBands, glb.fs)
	return power[2]

def power4a(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[3], frequencyBands, glb.fs)
	return power[0]

def power4b(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[3], frequencyBands, glb.fs)
	return power[1]

def power4c(X, channel):
	frequencyBands = [0.1, 4, 8, 12,30]
	power, powerRatio = pyeeg.bin_power(X[3], frequencyBands, glb.fs)
	return power[2]

def pearsonCoeff14(X, channel):
	pearsonCoefficients14 = np.corrcoef(X[0], X[3])
	return pearsonCoefficients14[1][0]

def pearsonCoeff14a(X, channel):
	pearsonCoefficients14 = np.corrcoef(X[0], X[3])
	return pearsonCoefficients14[0][1]

def pearsonCoeff13(X, channel):
	pearsonCoefficients13 = np.corrcoef(X[0], X[2])
	return pearsonCoefficients13[1][0]

def pearsonCoeff13a(X, channel):
	pearsonCoefficients13 = np.corrcoef(X[0], X[2])
	return pearsonCoefficients13[0][1]

def pearsonCoeff12(X, channel):
	pearsonCoefficients12 = np.corrcoef(X[0], X[1])
	return pearsonCoefficients12[1][0]

def pearsonCoeff12a(X, channel):
	pearsonCoefficients12 = np.corrcoef(X[0], X[1])
	return pearsonCoefficients12[0][1]

def pearsonCoeff34(X, channel):
	pearsonCoefficients12 = np.corrcoef(X[2], X[3])
	return pearsonCoefficients12[1][0]

def pearsonCoeff34a(X, channel):
	pearsonCoefficients12 = np.corrcoef(X[2], X[3])
	return pearsonCoefficients12[0][1]

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
	channel = 2
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
	dataset.setDatasetFolder(2)
	X, y = dataset.loadDataset(filename=datasetfile, filterCondition=True, 
                                filterType="DcNotch", removePadding=True, shift=False, windowLength=250)
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
	XLtrain, XLtest, yTrain, yTest, XL, scaler = classifier.scaleAndSplit(XL, y[0])
	#scaler = StandardScaler()
	#XL = scaler.fit_transform(XL, y[0])
	#XLtest = scaler.fit_transform(XLtest, yTest)


	clf = svm.SVC(kernel="linear", C = 10, decision_function_shape = 'ovr')
	#clf = svm.LinearSVC(penalty = 'l2',  loss='squared_hinge', dual = False, C = 10, random_state = 42)
	#clf = RandomForestClassifier(n_estimators = 45, max_depth = 10,  min_samples_leaf = 1, random_state = 40)
	#clf = svm.LinearSVC(penalty = 'l2', dual = False, C = 50, random_state = 42)
	#clf = linear_model.SGDClassifier(penalty = 'l2', random_state = 42)
	

	rfecv = RFECV(estimator=clf, step=1, cv=10, n_jobs=n_jobs,
	              scoring='accuracy')
	rfecv.fit(XL, y[0])

	print("Optimal number of features : %d" % rfecv.n_features_)
	print("Optimal features: ")
	print(rfecv.support_)

	writeFeatureMask(rfecv.support_)

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

	print("After feature selection: ")
	scores = cross_val_score(rfecv.estimator_, XLtrain, yTrain, cv=50, scoring = 'accuracy')
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	print()
	print("Scores")
	print(scores)

def writeFeatureMask(mask):	
	features = range(len(FUNC_MAP))
	featuremaskList = list(compress(features, mask))
	featuremask = open("featuremask.txt", 'w+')
	featuremask.write(str(featuremaskList))
	featuremask.close()

def readFeatureMask():
	featuremaskFile = open("featuremask.txt", 'r')
	featuremaskString = featuremaskFile.read()
	featuremaskFile.close()
	featuremaskString = featuremaskString[1:-1]
	#featuremaskString.pop(0)
	#featuremaskString.pop(-1)
	featuremaskList = map(int, featuremaskString.split(', '))
	print featuremaskList
	return featuremaskList



def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def compareFeatures(n_jobs=1):
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
	print("Is this a debug session? [Y/n]")
	inputString = raw_input()
	if inputString == "Y":
		debug = True
		print("Debug session activated, results will not be valid.")
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
	X, y = dataset.loadDataset(filename="data.txt", filterCondition=True, 
                                filterType="DcNotch", removePadding=True, shift=True, windowLength=100)
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
	#XLtrain, XLtest, yTrain, yTest = classifier.scaleAndSplit(XL, y[0])
	XLtrain, XLtest, yTrain, yTest, XL, scaler = classifier.scaleAndSplit(XL, y[0])
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

				#Append scores
				allPermutations.append(p)
				allParams.append(bestParams)
				allP.append(presc)
				allPavg.append(np.average(presc, weights=s))
				allR.append(r)
				allF1.append(f1)
				allS.append(s)

				if logging:
					logfile = open(dir_path+slash+"Logs"+slash+"Logfile.txt", 'a+')
					logfile.write("Feature combination:")
					logfile.write(str(p)+"\n")
					logfile.write(report+"\n\n\n")
					logfile.close()
	
					permfile = open(dir_path+slash+"Logs"+slash+"PermutationLog"+ str(i)+".txt", 'a+')
					permfile.write(":")
					permfile.write(str(p))
					permfile.close()
	
					permfile = open(dir_path+slash+"Logs"+slash+"ParameterLog"+ str(i)+".txt", 'a+')
					permfile.write(";")
					permfile.write(str(bestParams))
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

					permfile = open(dir_path+slash+"Logs"+slash+"SupportLog"+ str(i)+".txt", 'a+')
					permfile.write(":")
					for k in range(len(s)):
						permfile.write(','+str(s[k]))
					permfile.close()

				if debug:
					winner = allPavg.index(max(allPavg)) #Check for max average precision
					print(report)
					print("Best features so far are: " + convertPermutationToFeatureString(allPermutations[winner]))
					print("Best result so far are: ", allPavg[winner])
					
				#print("Pest parameters for this feature combination: " + str(bestParams))
				stop = datetime.now()
				numberOfCombinations -= 1
				trainings += 1
				remainingTime = (stop-start)*numberOfCombinations
				elapsedTime = (stop-start)
				print("Training number: %d" %trainings)
				print("Remaining combinations: %d" %numberOfCombinations)
				print("Elapsed time for training with this combination: " + str(elapsedTime))
				print("Estimated remaining time: " + str(remainingTime))

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

	if sendMail:
	    mail.sendemail(from_addr    = 'dronemasterprosjekt@gmail.com',
						to_addr_list = ['krishk@stud.ntnu.no','adriari@stud.ntnu.no'],
						cc_addr_list = [],
						subject      = "Training finished with combinations of %d to %d features" %(minNumFeatures, maxNumFeatures),
						message      = "Best result is with these features: "+str(allPermutations[winner]) + "\n"
										+ classification_report(yTest, yPred),
						login        = 'dronemasterprosjekt',
						password     = 'drone123')



def readLogs(length):
    import ast
    permfile = open(dir_path+slash+"Logs"+slash+"PermutationLog"+ str(length)+".txt", 'r')
    PermutationsString = permfile.read()
    permfile.close()
    PermutationsList = PermutationsString.split(':')
    PermutationsList.pop(0)
    #PermutationsList = tuple(PermutationsList)
    #Might need some more processing, now returns a list of tuples
    #print PermutationsList[28]
    for i in range(len(PermutationsList)):
		#print(eval(PermutationsList[i]))
		#PermutationsList[i] = tuple(map(int, PermutationsList[i][1:-1].split(',')))
		PermutationsList[i] = tuple(eval(PermutationsList[i]))


    permfile = open(dir_path+slash+"Logs"+slash+"ParameterLog"+ str(length)+".txt", 'r')
    ParametersString = permfile.read()
    permfile.close()
    ParametersList = ParametersString.split(';')
    ParametersList.pop(0)
    ParametersList = [ast.literal_eval(i) for i in ParametersList]

    permfile = open(dir_path+slash+"Logs"+slash+"PrecisionLog"+ str(length)+".txt", 'r')
    PrecisionString = permfile.read()
    permfile.close()
    PrecisionList = PrecisionString.split(":")
    PrecisionList.pop(0)

    for j in range(len(PrecisionList)):
        PrecisionSubList = PrecisionList[j].split(',')
        PrecisionSubList.pop(0)
        PrecisionList[j] = list([float(i) for i in PrecisionSubList])

    permfile = open(dir_path+slash+"Logs"+slash+"RecallLog"+ str(length)+".txt", 'r')
    RecallString = permfile.read()
    permfile.close()
    RecallList = RecallString.split(":")
    RecallList.pop(0)

    for j in range(len(RecallList)):
        RecallSubList = RecallList[j].split(',')
        RecallSubList.pop(0)
        RecallList[j] = list([float(i) for i in RecallSubList])

    permfile = open(dir_path+slash+"Logs"+slash+"F1Log"+ str(length)+".txt", 'r')
    f1String= permfile.read()
    permfile.close()
    f1List = f1String.split(":")
    f1List.pop(0)

    for j in range(len(f1List)):
        f1SubList = f1List[j].split(',')
        f1SubList.pop(0)
        f1List[j] = list([float(i) for i in f1SubList])

    permfile = open(dir_path+slash+"Logs"+slash+"SupportLog"+ str(length)+".txt", 'r')
    supportString = permfile.read()
    permfile.close()
    supportList = supportString.split(":")
    supportList.pop(0)

    for j in range(len(supportList)):
        supportSubList = supportList[j].split(',')
        supportSubList.pop(0)
        supportList[j] = list([float(i) for i in supportSubList])

    return PermutationsList, ParametersList, PrecisionList, RecallList, f1List, supportList

def extractList(words, size):
	return [word for word in words if len(word) == size]

def evaluateLogs(length, evaluationParam="maxminprecision"):
	print("Evaluating " + evaluationParam)
	print("Start to read logs")
	PermutationsList, ParametersList, PrecisionList, RecallList, f1List, supportList = readLogs(length)
	print("Finished reading logs, logs contain %d elements" %len(PermutationsList))
	#minLengthFeatures = min(PermutationsList, key=len)
	#maxLengthFeatures = max(PermutationsList, key=len)
	#print minLengthFeatures
	#print maxLengthFeatures
	#print("Logs contain combinations of %d to %d features" %(minLengthFeatures, maxLengthFeatures))
	winner = None

	'''		
	for length in range(minLengthFeatures, maxLengthFeatures+1):
		PrecisionList = extractList(PrecisionList, length)
	'''
	
	if evaluationParam == "averageprecision":
		allPavg = []
		for i in range(len(PrecisionList)):
			allPavg.append(np.average(PrecisionList[i], weights=supportList[i]))
		winner = allPavg.index(max(allPavg))
	elif evaluationParam == "maxminprecision":
		allPmin = []
		for i in range(len(PrecisionList)):
			allPmin.append(min(PrecisionList[i]))
		winner = allPmin.index(max(allPmin))
	elif evaluationParam == "maxminrecall":
		allRmin = []
		for i in range(len(RecallList)):
			allRmin.append(min(RecallList[i]))
		winner = allRmin.index(max(allRmin))
	else:
		print("Invalid input")
		return -1

	print("Winner index: %d" %winner)
	print("Best features are: " + convertPermutationToFeatureString(PermutationsList[winner]))
	print("Best parameters are: " + str(ParametersList[winner]))
	print("Precision: " + str(PrecisionList[winner]))
	print("Recall: " + str(RecallList[winner]))
	writeFeatureMask(PermutationsList[winner])

	return PermutationsList[winner], ParametersList[winner]

def cleanLogs():
    logfile = open(dir_path+slash+"Logs"+slash+"Logfile.txt", 'w')
    logfile.truncate(0)
    logfile.close()

    permfile = open(dir_path+slash+"Logs"+slash+"PermutationLog.txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open(dir_path+slash+"Logs"+slash+"ParameterLog.txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open(dir_path+slash+"Logs"+slash+"PrecisionLog.txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open(dir_path+slash+"Logs"+slash+"RecallLog.txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open(dir_path+slash+"Logs"+slash+"F1Log.txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open(dir_path+slash+"Logs"+slash+"SupportLog.txt", 'w')
    permfile.truncate(0)
    permfile.close()


def main():
	#cleanLogs()
	compareFeatures2(n_jobs=-1)
	#compareFeatures(-1)
	#evaluateLogs(12, "averageprecision")
if __name__ == '__main__':
	main()
