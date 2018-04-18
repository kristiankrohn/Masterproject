import numpy as np

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
				PermutationsList[i] = tuple(eval(PermutationsList[i]))
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

def evaluateLogs(length, evaluationParam="maxminprecision"):
	print("Evaluating " + evaluationParam)
	print("Start to read logs")
	PermutationsList, PrecisionList, RecallList, f1List = readLogs(length)
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
	else:
		print("Invalid input")
		return -1

	print("Winner index: %d" %winner)
	print("Best features are: " + convertPermutationToFeatureString(PermutationsList[winner]))
	#print("Best parameters are: " + str(ParametersList[winner]))
	print("Average precision: " + str(np.average(PrecisionList[winner])))
	print("Precision: " + str(PrecisionList[winner]))
	print("Recall: " + str(RecallList[winner]))
	writeFeatureMask(PermutationsList[winner], "BruteForce"+evaluationParam+str(length))

	return PermutationsList[winner]

def cleanLogs(num):

    logfile = open(dir_path+slash+"Logs"+slash+"Logfile"+str(num)+".txt", 'w')
    logfile.truncate(0)
    logfile.close()

    permfile = open(dir_path+slash+"Logs"+slash+"PermutationLog"+str(num)+".txt", 'w')
    permfile.truncate(0)
    permfile.close()

    permfile = open(dir_path+slash+"Logs"+slash+"ParameterLog"+str(num)+".txt", 'w')
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
	evaluateLogs(9, "averageprecision")
if __name__ == '__main__':
	main()
