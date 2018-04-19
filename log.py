from globalconst import  *
import globalvar as glb
import numpy as np
import features
import executiontime
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def readLogs(length):
    import ast
    permfile = open(dir_path+slash+"Logs"+slash+"PermutationLog"+ str(length)+".txt", 'r')
    PermutationsString = permfile.read()
    permfile.close()
    PermutationsList = PermutationsString.split(':')
    PermutationsList.pop(0)
    #print(PermutationsList[50388])
    #print(PermutationsList[50389])
    #print(PermutationsList[50390])
    #print(len(PermutationsList[0]))
    #PermutationsList = tuple(PermutationsList)
    #Might need some more processing, now returns a list of tuples
    #print PermutationsList[28]
    for i in range(len(PermutationsList)):
		#print(eval(PermutationsList[i]))
		#print(i)
		#PermutationsList[i] = tuple(map(int, PermutationsList[i][1:-1].split(',')))
		var = PermutationsList[i]
		#print var
		var = var.replace('(', ',').replace(')', ',')
		#print var
		numlist = var.split(',')
		#print numlist
		numlist[:] = (value for value in numlist if value != '')
		#print numlist
		num = [int(q) for q in numlist]
		#print(len(num))
		if len(num) == length:
			try:
				featuremaskString = PermutationsList[i][1:-1]
				PermutationsList[i] = map(int, featuremaskString.split(', '))
				#PermutationsList[i] = tuple(eval(PermutationsList[i]))
			except:
				print("Unhandeled error")
				print(PermutationsList[i])
				print("Index = %d" %i)
		else:
			print(PermutationsList[i])
			print("Index = %d" %i)

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



    return PermutationsList, PrecisionList, RecallList, f1List

def extractList(words, size):
	return [word for word in words if len(word) == size]

def fixLogs(length):
	print("Start to read logs")
	PermutationsList, PrecisionList, RecallList, f1List = readLogs(length)
	print("Finished reading logs")

	print("Length of logs:")
	permlength = len(PermutationsList)
	precisionlength = len(PrecisionList)
	recalllength = len(RecallList)
	f1length = len(f1List)
	print("Permutations: %d" %permlength)
	print("Precision: %d" %precisionlength)
	print("Recall: %d" %recalllength)
	print("f1: %d" %f1length)
	expectedResults = features.nCr(26,length)
	print("Expected logs to be of length: %d" %expectedResults)
	
	c = Counter(map(tuple,PermutationsList))
	dups = [k for k,v in c.items() if v>1]
	numDups = len(dups)
	print("Number of duplicate combinations: %d" %numDups)

	missingComb = (permlength-numDups) - expectedResults
	print("Number of missing combinations: %d" %missingComb)
	print()
	print("Summary: ")
	if numDups > 0:		
		if missingComb == 0:
			print("Logs are complete, but contain duplicates")
		else:
			print("Logs not complete and contain duplicates")
	if numDups == 0:
		if missingComb == 0:
			print("Logs are complete")
			
def evaluateLogs(length, evaluationParam="average", metric="precision", energy="high"):
	print("Evaluating " + evaluationParam + " " + metric + " " + energy + " energy")
	print("Start to read logs")
	PermutationsList, PrecisionList, RecallList, f1List = readLogs(length)
	print("Finished reading logs, logs contain %d elements" %len(PermutationsList))
	#minLengthFeatures = min(PermutationsList, key=len)
	#maxLengthFeatures = max(PermutationsList, key=len)
	#print minLengthFeatures
	#print maxLengthFeatures
	#print("Logs contain combinations of %d to %d features" %(minLengthFeatures, maxLengthFeatures))
	winner = None
	print("Length of logs:")
	print(len(PermutationsList))
	print(len(PrecisionList))
	print(len(RecallList))
	print(len(f1List))


	'''
	for length in range(minLengthFeatures, maxLengthFeatures+1):
		PrecisionList = extractList(PrecisionList, length)
	'''

	if metric == "precision":
		metricList = PrecisionList
	elif metric == "recall":
		metricList = RecallList
	elif metric == "f1":
		metricList = f1List
	else:
		print("Invalid fuction parameters")
		return -1

	allList = []
	if evaluationParam == "average":
		#allAvg = []
		
		winner = allList.index(max(allList))

	elif evaluationParam == "maxmin":
		#allMin = []
		for i in range(len(metricList)):
			allList.append(min(metricList[i]))
		winner = allList.index(max(allList))	
	
	elif evaluationParam == "min": #Need to have a worst case to show the benefits of the brute force search
		for i in range(len(metricList)):
			allList.append(np.average(metricList[i]))
		winner = allList.index(min(allList))
		print("Worst features are: " + features.convertPermutationToFeatureString(PermutationsList[winner]))
		print("Featuremask: " + str(PermutationsList[winner]))
		print("Worst case avg precision: " + str(np.average(PrecisionList[winner])))
		print("Worst case precision: " + str(PrecisionList[winner]))
		print("Worst case recall: " + str(RecallList[winner]))
		return #Doen't need this result to anything yet
	elif evaluationParam =="plot":
		for i in range(len(metricList)):
			allList.append(np.average(metricList[i]))
		title = metric + " distribution"
		ax = sns.kdeplot(allList, shade=True)
		ax.set_title(title)
		plt.xlabel(metric)
		plt.ylabel("Density")
		plt.show()
		return
	else:
		print("Invalid evaluation parameters")
		return -1

	if energy == "low":

		maximum = max(allList)
		threshold = maximum - 0.01
		executionTimeList = executiontime.readExecutionTime(filename = 'executionTimeAllFeatures100.txt')
		costs = []
		candidates = []

		for j in range(len(allList)):
			if allList[j] >= threshold:
				costs.append(executiontime.getExecutionCost(PermutationsList[j], executionTimeList))
				candidates.append(j)
		print("Number of candidates with high enough accuracy: %d" %len(candidates))
		winnerCostIndex = costs.index(min(costs))
		winner = candidates[winnerCostIndex]
		
	'''
	if evaluationParam == "averageprecision":
		allPavg = []
		for i in range(len(PrecisionList)):
			allPavg.append(np.average(PrecisionList[i]))
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
	
	elif evaluationParam == "lowenergyaverageprecision":
		allPavg = []
		for i in range(len(PrecisionList)):
			allPavg.append(np.average(PrecisionList[i]))
		#winner = allPavg.index(max(allPavg))

		maximum = max(allPavg)
		threshold = maximum - 0.01
		executionTimeList = executiontime.readExecutionTime(filename = 'executionTimeAllFeatures100.txt')
		costs = []
		candidates = []

		for j in range(len(allPavg)):
			if allPavg[j] >= threshold:
				costs.append(executiontime.getExecutionCost(PermutationsList[j], executionTimeList))
				candidates.append(j)

		winnerCostIndex = costs.index(min(costs))
		winner = candidates[winnerCostIndex]

	else:
		print("Invalid input")
		return -1
	'''

	print("Winner index: %d" %winner)
	print("Best features are: " + features.convertPermutationToFeatureString(PermutationsList[winner]))
	print("Featuremask: " + str(PermutationsList[winner]))
	#print("Best parameters are: " + str(ParametersList[winner]))
	print("Average precision: " + str(np.average(PrecisionList[winner])))
	print("Precision: " + str(PrecisionList[winner]))
	print("Recall: " + str(RecallList[winner]))
	features.writeFeatureMask(PermutationsList[winner], "BruteForce"+evaluationParam+metric+energy+str(length))

	return PermutationsList[winner]

def cleanLogs(num):

    permfile = open(dir_path+slash+"Logs"+slash+"PermutationLog"+str(num)+".txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open(dir_path+slash+"Logs"+slash+"PrecisionLog"+str(num)+".txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open(dir_path+slash+"Logs"+slash+"RecallLog"+str(num)+".txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open(dir_path+slash+"Logs"+slash+"F1Log"+str(num)+".txt", 'w')
    permfile.truncate(0)
    permfile.close()

def main():

	#cleanLogs()
	#compareFeatures2(n_jobs=-1)
	#compareFeatures(-1)
	#readLogs(12)
	#evaluateLogs(9, evaluationParam="min", metric="precision", energy="high")
	#evaluateLogs(10, evaluationParam="maxmin", metric="recall", energy="low")
	#evaluateLogs(9, evaluationParam="plot", metric="precision", energy="high")
	fixLogs(10)
if __name__ == '__main__':
	main()
