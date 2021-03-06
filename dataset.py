import os, shutil
import threading
from globalconst import*
import globalvar as glb
from operator import sub
import plot
import matplotlib.pyplot as plt
import filterlib
import time as tme
import copy
import multiprocessing
import itertools
from scipy import signal
printlock = Lock()
filelock = Lock()



frontPadding = 750
backPadding = 250
#Datalist = []
DataSet = []

def saveShortTemp(direction):
	global filelock, printlock
	f = prepSaveData(direction, shortLength)

	if f != -1:
		with filelock:
			file = open(dir_path+glb.datasetFolder+"temp.txt", 'a')
			file.write(f)
			file.close()
		#print("Save short temp completed")
	else:
		print("Failed to save short temp")

def saveLongTemp(direction):
	global filelock, printlock
	f = prepSaveData(direction, longLength)

	if f != -1:
		with filelock:
			file = open(dir_path+glb.datasetFolder+"longtemp.txt", 'a')
			file.write(f)
			file.close()
		#print("Save long temp completed")
	else:
		print("Failed to save long temp")

def prepSaveData(direction, length):
	global mutex, filelock
	#global printlock
	if direction == 5:
		delayconstant = 0.2
	else:
		if length == shortLength:
			delayconstant = 0.8
		elif length == longLength:
			delayconstant = 0.6

	startTime = tme.time() + (backPadding/glb.fs) + delayconstant
	ready = False

	temp = None


	while not ready:
		with glb.mutex:
			if len(glb.data[0][timestamp]) < (length + frontPadding + backPadding):
				print("Lenght of data is too small")
				return -1
			elif glb.data[0][timestamp][-1] >= startTime: #Gaar out of range
				ready = True
				temp = copy.deepcopy(glb.data)
				error = False
				for i in range(numCh):

					if temp[i][timestamp][-1] != temp[i-1][timestamp][-1]:
						print("Timestamp error, unequal for same index")
						return -1
					if len(temp[i][rawdata]) != len(temp[i-1][rawdata]):
						print("Uneven length of rawdata")
						return -1
					if len(temp[i][filterdata]) != len(temp[i-1][filterdata]):
						print("Uneven length of filterdata")
						return -1
					if len(temp[i][timestamp]) != len(temp[i-1][timestamp]):
						print("Uneven length of timestamp")
						return -1
					if len(temp[i][timestamp]) != len(temp[i][rawdata]):
						print("Uneven length between timestamp and rawdata")
						return -1


	stopindex = len(temp[0][rawdata])-1


	for i in range((len(temp[0][timestamp])-1), 0, -1):
		#print(i)
		if temp[0][timestamp][i]<=startTime:
			stopindex = i
			break
		elif i == (length + frontPadding + backPadding):
			print("Index error, could not find stop index")
			return -1

	stop = stopindex
	start = stop - (length + frontPadding + backPadding)


	f = ""
	for j in range(numCh):

		if direction >= 10:
			print("Unknown direction")
			return -1
		else:
			f += directioncode[direction]
			f += str(j) #Kanal

			for i in range(start, stop):
				f += ","
				try:
					num = temp[j][rawdata][i]

				except IndexError:
					print("Index error")
					print("Len buffer = %d" %(len(temp[j][rawdata])-1))
					print("Index = %d" % i)
					print("Write operation aborted")

					return -1

				f += str(num)
			f += ":"

	return f


def exportPlots(command, plottype="time", speed="slow"):
	global filelock
	global DataSet

	#folder = dir_path + "\\Dataset_exports\\figures\\Center"
	#print(folder)
	if command == "temp":
		folders = [slash + "tempfigures", slash + "longtempfigures"]

	elif command == "data":
		folders = [slash + "figures", slash + "longfigures"]

	else:
		folders = [slash + "figures", slash + "tempfigures",
				slash + "longfigures", slash + "longtempfigures"]

	if speed != "slow":
		multiprocessing.freeze_support()

	for i in range(len(folders)):

		for j in range(len(movements)):
			if plottype == "fft":
				folder = dir_path + slash + "Dataset_fft" + folders[i] + movements[j]
			elif plottype == "time":
				folder = dir_path + slash +  "Dataset_exports" + folders[i] + movements[j]
			elif plottype == "raw":
				folder = dir_path + slash + "Dataset_raw" + folders[i] + movements[j]
			else:
				print("Invalid plottype")
				return


			for the_file in os.listdir(folder): #Removes all files in folders
				file_path = os.path.join(folder, the_file)
				try:
					if os.path.isfile(file_path):
						os.unlink(file_path)
		        	#elif os.path.isdir(file_path): shutil.rmtree(file_path)
				except Exception as e:
					print(e)
					return

		file = None

		if command == "temp":
			if i == 0:
				file = open(dir_path+glb.datasetFolder+"temp.txt", 'r')
			else:
				file = open(dir_path+glb.datasetFolder+"longtemp.txt", 'r')
		elif command == "data":
			if i == 0:
				file = open(dir_path+glb.datasetFolder+"data.txt", 'r')
			else:
				file = open(dir_path+glb.datasetFolder+"longdata.txt", 'r')
		else:

			if i == 0:
				file = open(dir_path+glb.datasetFolder+"data.txt", 'r')
			elif i == 1:
				file = open(dir_path+glb.datasetFolder+"temp.txt", 'r')
			elif i == 2:
				file = open(dir_path+glb.datasetFolder+"longdata.txt", 'r')
			elif i == 3:
				file = open(dir_path+glb.datasetFolder+"longtemp.txt", 'r')


		with filelock:
			AllData = file.read()
			file.close()

		#DataSet = []
		DataSet = AllData.split(':')
		'''
		if (speed == "fast") and ((len(DataSet)/numCh) > 150):
			print("Size of dataset is: %d" %(len(DataSet)/numCh) + ", do you still want to plot fast?(this might halt your cpu) [Y/n]")
			inputString = raw_input()
			if inputString != "Y":
				print("Changed to slow sequential plot export")
				speed = "slow"
		'''
		print("Number of plots: %d" %(len(DataSet)/numCh))
		if speed == "fast":
			if len(DataSet) < numCh:
				return
			else:

				split = 100
				splitsize = numCh*split
				numSplits = len(DataSet)//splitsize
				if len(DataSet)%splitsize != 0:
					numSplits += 1

				for k in range(numSplits):
					if k < (numSplits-1):
						DataSetTemp = DataSet[k*splitsize:(k+1)*splitsize]
					elif k == (numSplits-1):
						DataSetTemp = DataSet[k*splitsize:-1]
					num_cpu = multiprocessing.cpu_count()

					pool = multiprocessing.Pool(len(DataSetTemp)/numCh)
					variables = [None]*4
					#variables[0] = 0
					variables[0] = i
					variables[1] = command
					variables[2] = plottype
					variables[3] = 0
					#variables[4] = DataSet
					#variables[5] = 0
					indexlist = range(len(DataSetTemp)/numCh)
					for m in range(len(indexlist)):
						indexlist[m] += k*split
					Datalist = [None]*(len(DataSetTemp)/numCh)
					#print(indexlist)
					for g in range(len(DataSetTemp)/numCh):
						start = g*numCh
						stop = (g+1)*numCh
						#print("Start: %d"%start)
						#print("Stop: %d"%stop)
						#print("g: %d"%g)
						#if len(DataSet) <= stop:
							#break
						#Datalist.append(DataSet[start:stop])
						variables[3] = DataSetTemp[start:stop]

						#variables[4] = str(g)
						Datalist[g]=copy.deepcopy(variables)
						#indexlist.append(g)
					#Datalist = [None]*(len(DataSet)/numCh)
					#for i in range(len(Datalist)):
						#print(Datalist[i])
						#tme.sleep(1)
					#Dataset = None
					print("Spawning %d threads" %(len(DataSetTemp)/numCh))
					#threadexport = [None]*(len(DataSet)/numCh)
					#print("Thread index up to: %d" %len(DataSet))
					#print(i)

					#res = pool.map(func_star, itertools.izip(indexlist,itertools.repeat(variables)), num_cpu)
					iterator = itertools.izip(indexlist,Datalist)
					#Datalist = None
					#indexlist = None
					res = pool.map(func_star, iterator, num_cpu)
					res = [r for r in res if r is not None]
					#iterator = None
					#print(p)
					pool.close()
					print("Waiting to join")
					pool.join()
					print("Has joined")

		else:

			for k in range(0, len(DataSet), numCh):
				care = True
				feature = []
				feature = DataSet[k].split(',')

				if len(feature) > 10:
					featuretype = feature[0]
					feature.pop(0)

					if featuretype[0] == 'u':
						title = "Up"
					elif featuretype[0] == 'd':
						title = "Down"
					elif featuretype[0] == 'l':
						title = "Left"
					elif featuretype[0] == 'r':
						title = "Right"
					elif featuretype[0] == 'c':
						title = "Center"
						#care = False
					else:
						care = False

					if featuretype[1] == 'd':
						title += "_direction"
					elif featuretype[1] == 'r':
						title += "_return"
					elif featuretype[1] == 's':
						title += "_still"
					elif featuretype[1] == 'b':
						title += "_blink"
					else:
						title = "Garbage"
						#care = False

					if care:

						plt.figure(figsize=(20,10))
						plt.suptitle(title)

						b, a = filterlib.designfilter()
						for l in range(0, numCh):

							feature1 = DataSet[k+l].split(',')
							featuretype1 = feature1[0]
							feature1.pop(0)

							if not(featuretype[0] == featuretype1[0]):
								break

							featureData1 = map(float, feature1)
							subplotnum = (numCh/2)*100 + 20 + l + 1

							ax1 = plt.subplot(subplotnum)
							if plottype != "raw":

								featureData1 = filterlib.plotfilter(featureData1, b, a)

								if l == 0:
									peakIndex = findPeakIndex(featureData1)
									#featureData1[frontPadding + peakIndex]=200
								if len(featureData1) > 1600:
									featureData1 = featureData1[frontPadding:-backPadding] #Remove paddings
									print("Statically remove padding")
								else: #Short set
									featureData1 = rmPadding(featureData1, peakIndex)
									#pass
									#print("Dynamicaly remove padding with centering")
							if plottype == "fft":
								plot.exportFftPlot(featureData1, channels[l], ax1)
							elif plottype == "raw":
								plot.exportRaw(featureData1, channels[l], ax1)
							else:
								plot.exportplot(featureData1, channels[l], ax1)


						tempOrNot = None
						#print("i = %d" %i)
						#print("Command" + command)
						if command == "temp":
							if i == 0:
								tempOrNot = "tempfigures"
							else:
								tempOrNot = "longtempfigures"
						elif command == "data":
							if i == 0:
								tempOrNot = "figures"
							else:
								tempOrNot = "longfigures"
						else:
							if i == 0:
								tempOrNot = "figures"
							elif i == 1:
								tempOrNot = "tempfigures"
							elif i == 2:
								tempOrNot = "longfigures"
							elif i == 3:
								tempOrNot = "longtempfigures"

						if plottype == "fft":
							savestring = (dir_path + slash + "Dataset_fft" + slash
											+ tempOrNot + slash + title + slash
											+ title+str(k/numCh) + ".png")
						elif plottype == "raw":
							savestring = (dir_path + slash + "Dataset_raw" + slash
											+ tempOrNot + slash + title + slash
											+ title+str(k/numCh) + ".png")
						else:
							savestring = (dir_path + slash + "Dataset_exports" + slash
											+ tempOrNot +slash + title + slash
											+ title+str(k/numCh) + ".png")
						print(savestring)

						plt.subplots_adjust(hspace=0.6)
						with filelock:
							plt.savefig(savestring, bbox_inches='tight')

						#plt.show()
						plt.close()

				else:
					print("Empty file")

	#plt.close('all')

	print("Finished exporting plots")

def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return threadplots(*a_b)


#def threadplots(k, DataSet, numCh, command, plottype):
def threadplots(k, variables):
	#DataSet
	import matplotlib.pyplot as plt
	#print(len(variables))

	#channels = variables[0]
	i = variables[0]
	command	= variables[1]
	plottype = variables[2]
	DataSet = variables[3]
	#p = variables[5]
	#k = p
	#k = 6
	#print("Thread %d" %k)
	#print("K = %d" %k)
	#print(Datalist[k][0])
	care = True
	feature = []
	#feature = DataSet[k*numCh].split(',')
	feature = DataSet[0].split(',')
	#print("First index error")
	if len(feature) > 10:
		featuretype = feature[0]
		feature.pop(0)

		if featuretype[0] == 'u':
			title = "Up"
		elif featuretype[0] == 'd':
			title = "Down"
		elif featuretype[0] == 'l':
			title = "Left"
		elif featuretype[0] == 'r':
			title = "Right"
		elif featuretype[0] == 'c':
			title = "Center"
			#care = False
		else:
			care = False

		if featuretype[1] == 'd':
			title += "_direction"
		elif featuretype[1] == 'r':
			title += "_return"
		elif featuretype[1] == 's':
			title += "_still"
		elif featuretype[1] == 'b':
			title += "_blink"
		else:
			title = "Garbage"
			#care = False

		if care:

			plt.figure(figsize=(20,10))
			#print(title)
			plt.suptitle(title)
			#print("Penis2")
			b, a = filterlib.designfilter()
			for l in range(0, numCh):

				#feature1 = DataSet[k*numCh+l].split(',')
				feature1 = DataSet[l].split(',')
				#print("Second indexerror")
				featuretype1 = feature1[0]
				feature1.pop(0)

				if not(featuretype[0] == featuretype1[0]):
					break

				featureData1 = map(float, feature1)
				subplotnum = (numCh/2)*100 + 20 + l + 1

				ax1 = plt.subplot(subplotnum)
				if plottype != "raw":
					featureData1 = filterlib.plotfilter(featureData1, b, a)
					featureData1 = featureData1[frontPadding:-backPadding] #Remove paddings
				if plottype == "fft":
					plot.exportFftPlot(featureData1, channels[l], ax1)
				elif plottype == "raw":
					plot.exportRaw(featureData1, channels[l], ax1)
				else:
					plot.exportplot(featureData1, channels[l], ax1)

			#print("Penis3")
			tempOrNot = None
			#print("i = %d" %i)
			#print("Command" + command)
			if command == "temp":
				if i == 0:
					tempOrNot = "tempfigures"
				else:
					tempOrNot = "longtempfigures"
			elif command == "data":
				if i == 0:
					tempOrNot = "figures"
				else:
					tempOrNot = "longfigures"
			else:
				if i == 0:
					tempOrNot = "figures"
				elif i == 1:
					tempOrNot = "tempfigures"
				elif i == 2:
					tempOrNot = "longfigures"
				elif i == 3:
					tempOrNot = "longtempfigures"

			if plottype == "fft":
				savestring = (dir_path + slash + "Dataset_fft" + slash
								+ tempOrNot + slash + title + slash
								+ title+str(k) + ".png")
			elif plottype == "raw":
				savestring = (dir_path + slash + "Dataset_raw" + slash
								+ tempOrNot + slash+title + slash
								+ title+str(k) + ".png")
			else:
				savestring = (dir_path + slash + "Dataset_exports" + slash
								+ tempOrNot +slash+title +slash
								+ title+str(k) + ".png")
			#print(savestring)

			plt.subplots_adjust(hspace=0.6)
			#with filelock:
			plt.savefig(savestring, bbox_inches='tight')

			#plt.show()
			plt.close()

	else:
		print("Empty file")
	print("Finished plot: %d" %k)


def saveShortData():
	print("Are you sure you want to append all elements in shorttemp to shortdata?")
	print("Continue? [Y/n]")
	inputString = raw_input()
	if inputString == "Y":
		tempfile = open(dir_path+glb.datasetFolder+"temp.txt", 'r')
		tempData = tempfile.read()
		tempfile.close()
		permfile = open(dir_path+glb.datasetFolder+"data.txt", 'a')
		permfile.write(tempData)
		permfile.close()
		tempfile = open(dir_path+glb.datasetFolder+"temp.txt", 'w')
		tempfile.truncate(0)
		tempfile.close()
		print("Short Data Saved")
	else:
		print("Append operation aborted")


def saveLongData():
	print("Are you sure you want to append all elements in longtemp to longdata?")
	print("Continue? [Y/n]")
	inputString = raw_input()
	if inputString == "Y":
		tempfile = open(dir_path+glb.datasetFolder+"longtemp.txt", 'r')
		tempData = tempfile.read()
		tempfile.close()
		permfile = open(dir_path+glb.datasetFolder+"longdata.txt", 'a')
		permfile.write(tempData)
		permfile.close()
		tempfile = open(dir_path+glb.datasetFolder+"longtemp.txt", 'w')
		tempfile.truncate(0)
		tempfile.close()
		print("Long Data Saved")
	else:
		print("Append operation aborted")




def clear(elementtype):
	if elementtype == "shorttemp":
		targetfile = "temp.txt"
	elif elementtype == "longtemp":
		targetfile = "longtemp.txt"
	elif elementtype == "shortdata":
		targetfile = "data.txt"
	elif elementtype == "longdata":
		targetfile = "longdata.txt"
	else:
		print("Error: wrong elementtype")
		return
	print("Are you sure you want to delete all elements in: " + elementtype)
	print("Continue? [Y/n]")
	inputString = raw_input()
	if inputString == "Y":
		tempfile = open(dir_path+glb.datasetFolder+targetfile, 'w')
		tempfile.truncate(0)
		tempfile.close()
		print("All data in " + elementtype + " are deleted")
	else:
		print("Delete operation aborted")



def loadDataset(filename="data.txt", filterCondition=True, filterType="DcNotch", removePadding=True, shift=True, windowLength = 250):
	global filelock
	print("Starting to load dataset")
	x = [[],[],[],[],[],[],[],[]]
	y = [[],[],[],[],[],[],[],[]]

	file = None
	print(dir_path+glb.datasetFolder+filename)
	with filelock:
		file = open((dir_path+glb.datasetFolder+filename), 'r')
		AllData = file.read()
		file.close()

	DataSet = []
	DataSet = AllData.split(":")
	#print(DataSet)
	if filterType == "DC":
		a = [1 , -0.9]
		b = [1,-1]
	else:
		b, a = filterlib.loadfilter()
		if b.all() == None or a.all() == None:
			print("Designing filter from scratch")
			b, a = filterlib.designfilter()

	for k in range(0, len(DataSet)):
		#print("k = %d" %k)
		care = True
		feature = []
		feature = DataSet[k].split(",")
		#print("Length of vector: %d\n" %len(feature))
		if len(feature) > 10:
			featuretype = feature[0]
			label = featuretype[0:2]
			channel = map(int, featuretype[2])
			channel = channel[0]
			feature.pop(0)

			y[channel].append(directioncode.index(label))

			featureData = map(float, feature)
			#featureData = filterlib.plotfilter(featureData, b, a)
			#print("Raw data: %0.2f\n" %featureData[0])
			if filterCondition:
				featureData = filterlib.plotfilter(featureData, b, a)
			if removePadding:

				if (channel == 0) and shift and (len(featureData) < 1600):
					peakIndex = findPeakIndex(featureData, plot=False, label = y[channel])
				else: 
					peakIndex = (len(featureData) - frontPadding - backPadding) // 2

				featureData = rmPadding(featureData, peakIndex, windowLength)
					
			x[channel].append(featureData)

		else:
			print("Invalid datapoint")

	#plt.close('all')
	#print(len(x[0][0]))
	#print y
	print("Finished loading dataset")
	return (x, y)




def sortDataset(x=None, y=None, length=10, classes=[0,5,4,2,6,8], merge=False, zeroClassMultiplier=1):
	if (x==None) or (y==None):
		x, y = loadDataset()

	xReturn = [[],[],[],[],[],[],[],[]]
	yReturn = [[],[],[],[],[],[],[],[]]
	counts = [None]*len(classes)
	print("Statistics before sort: ")
	for i in range(len(classes)):
		print("Number of occurances of class %d:" %classes[i])
		counts[i] = y[0].count(classes[i])
		print(counts[i])
	minValElement = min(counts)
	print("Lowest occurance of a class is: %d" %minValElement)
	if minValElement == 0:
		print("List is empty or class cannot be found in set, returning empty arrays")
		return xReturn, yReturn

	elif minValElement < length:
		print("Given length is too long, max for this dataset is: %d" %minValElement)
		length = minValElement
	
	returnCounts = [0]*len(classes)
	
	
	if 0 in classes:
		zeroIndex = classes.index(0)
	else:
		zeroIndex = -1
		returnCounts.append(0)
	
	if 5 in classes:
		fiveIndex = classes.index(5)
	else:
		if zeroIndex == -1:
			fiveIndex = -2
		else:
			fiveIndex = -1
		returnCounts.append(0)
	

	for i in range(len(y[0])):
		if y[0][i] in classes:
			classitemindex = classes.index(y[0][i])
			#This will fail if 5 or 0 is not in classes
			if ((returnCounts[classitemindex] < length) 
				or (merge and(((classitemindex == zeroIndex) and (returnCounts[zeroIndex] < length*zeroClassMultiplier*2)) 
				or ((classitemindex == fiveIndex) and (returnCounts[fiveIndex] < length*zeroClassMultiplier*2))))): 
				
				returnCounts[classitemindex] += 1
				for j in range(numCh):
					xReturn[j].append(x[j][i])
					if merge:
						yReturn[j].append(MergeDict[y[j][i]])
					else:
						yReturn[j].append(y[j][i])




	if merge:
		x = copy.deepcopy(xReturn)
		y = copy.deepcopy(yReturn)
		NewClasses = []
		for i in range(len(classes)):
			NewClasses.append(MergeDict[classes[i]])
		classes = np.unique(NewClasses).tolist()
		xReturn = [[],[],[],[],[],[],[],[]]
		yReturn = [[],[],[],[],[],[],[],[]]
		counts = [None]*len(classes)

		#print("Statistics before sort: ")
		for i in range(len(classes)):
			#print("Number of occurances of class %d:" %classes[i])
			counts[i] = y[0].count(classes[i])
			#print(counts[i])
		minValElement = min(counts)
		#print("Lowest occurance of a class is: %d" %minValElement)

		returnCounts = [0]*len(classes)

		for i in range(len(y[0])):
			if y[0][i] in classes:
				classitemindex = classes.index(y[0][i])
				if y[0][i] == 0 or y[0][i] == 5:
					zeroclass = True
				else:
					zeroclass = False
				#This will fail if 5 or 0 is not in classes
				if ((returnCounts[classitemindex] < minValElement) or 
					((zeroclass == True) and (returnCounts[classitemindex] < (minValElement*(zeroClassMultiplier))))): 
					
					returnCounts[classitemindex] += 1
					for j in range(numCh):
						xReturn[j].append(x[j][i])
						yReturn[j].append(y[j][i])

	


	counts = [None]*len(classes)
	print("Statistics after sort: ")
	for i in range(len(classes)):
		print("Number of occurances of class %d:" %classes[i])
		counts[i] = yReturn[0].count(classes[i])
		print(counts[i])

	return (xReturn, yReturn)



def datasetStats(filename="data.txt"):
	x,y = loadDataset(filename)
	counts = [None]*len(longclasses)
	print("Statistics: ")
	for i in range(len(longclasses)):
		print("Number of occurances of class %d:" %longclasses[i])
		counts[i] = y[0].count(longclasses[i])
		print(counts[i])

def deletesystem(elementtype="shorttemp"):
	#Some system for setting variables according to file
	if elementtype == "shorttemp":
		targetfile = "temp.txt"
		file = "deletetemp.txt"
	elif elementtype == "longtemp":
		targetfile = "longtemp.txt"
		file = "deletelongtemp.txt"
	elif elementtype == "shortdata":
		targetfile = "data.txt"
		file = "deletedata.txt"
	elif elementtype == "longdata":
		targetfile = "longdata.txt"
		file = "deletelongdata.txt"
	else:
		print("Error: wrong elementtype")
		return

	tempfile = open(dir_path+ slash +"Dataset_delete" + slash +file, 'r')
	tempData = tempfile.read()
	tempfile.close()

	tempindexlist = []
	tempindexlist = tempData.split(',')
	tempindexlist.pop(-1)
	#print(tempindexlist)
	if len(tempindexlist) > 0:
		indexlist = map(int, tempindexlist)
		#print(indexlist)
		indexlist = np.unique(indexlist).tolist()
		#print(indexlist)
		indexlist.sort(reverse=True)
		print("Data to delete in " + elementtype + ": ")
		print(indexlist)
		print("Continue? [Y/n]")
		inputString = raw_input()
		if inputString == "Y":
			for i in range(len(indexlist)):
				#Delete data
				deleteelement(index = indexlist[i], filename=targetfile)
				#print(indexlist[i])
			print("Sucessfully deleted elements")
			tempfile = open(dir_path+slash+"Dataset_delete"+slash+file, 'w')
			tempfile.truncate(0)
			tempfile.close()
		else:
			print("Del operation aborted")
	else:
		print(elementtype + " list is empty")



def deleteelement(index, filename):

	index = index * numCh
	file = open(dir_path+glb.datasetFolder+filename, 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	if index > len(DataSet):
		print("Index error")
		print("Index = %d" %index)
		print("Length of dataset = %d" %len(DataSet))
	else:
		for i in range(numCh):
			DataSet.pop(index)
		#DataSet.pop(index)
		#print(DataSet)
		file.close()
		file = open(dir_path+glb.datasetFolder+filename, 'w')
		for i in range(len(DataSet)-1):
			file.write(DataSet[i])
			file.write(':')
		file.close()
		index = index / numCh
		print(filename +" element %d is deleted" % index)



def appenddelete(index, elementtype):
	if elementtype == "shorttemp":
		filename = "deletetemp.txt"
	elif elementtype == "longtemp":
		filename = "deletelongtemp.txt"
	elif elementtype == "shortdata":
		filename = "deletedata.txt"
	elif elementtype == "longdata":
		filename = "deletelongdata.txt"
	else:
		print("Error: wrong elementtype")
		return

	file = open(dir_path+slash+"Dataset_delete"+slash+filename, 'a')
	file.write(str(index))
	file.write(",")
	file.close()
	print("Appended %d to "%index + elementtype)



def remove_appenddelete(index, elementtype):
	if elementtype == "shorttemp":
		filename = "deletetemp.txt"
	elif elementtype == "longtemp":
		filename = "deletelongtemp.txt"
	elif elementtype == "shortdata":
		filename = "deletedata.txt"
	elif elementtype == "longdata":
		filename = "deletelongdata.txt"
	else:
		print("Error: wrong elementtype")
		return

	tempfile = open(dir_path+slash+"Dataset_delete"+slash+filename, 'r')
	tempData = tempfile.read()
	tempfile.close()

	tempindexlist = []
	tempindexlist = tempData.split(',')
	tempindexlist.pop(-1)
	if len(tempindexlist) > 0:
		indexlist = map(int, tempindexlist)
		indexlist = np.unique(indexlist).tolist()
		if index in indexlist:
			indexlist.remove(index)
			file = open(dir_path+slash+"Dataset_delete"+slash+filename, 'w')
			for i in range(len(indexlist)):
				file.write(str(indexlist[i]))
				file.write(",")
			file.close()
			print("Removed index %d  "%index + "from " + elementtype)
		else:
			print("Index does not exist in " + elementtype)
	else:
		print(elementtype + " list is empty!")



def print_appenddelete(elementtype):
	if elementtype == "shorttemp":
		filename = "deletetemp.txt"
	elif elementtype == "longtemp":
		filename = "deletelongtemp.txt"
	elif elementtype == "shortdata":
		filename = "deletedata.txt"
	elif elementtype == "longdata":
		filename = "deletelongdata.txt"
	else:
		print("Error: wrong elementtype")
		return

	tempfile = open(dir_path+slash+"Dataset_delete"+slash+filename, 'r')
	tempData = tempfile.read()
	tempfile.close()

	tempindexlist = []
	tempindexlist = tempData.split(',')
	tempindexlist.pop(-1)
	#print(tempindexlist)
	if len(tempindexlist) > 0:
		indexlist = map(int, tempindexlist)
		#print(indexlist)
		indexlist = np.unique(indexlist).tolist()
		#print(indexlist)
		indexlist.sort(reverse=True)
		print("Data to delete in " + elementtype + ": ")
		print(indexlist)
	else:
		print(elementtype + " list is empty")

def setDatasetFolder(folder):
	if folder == 1:
		glb.datasetFolder = datasetFolders[0]

	elif folder == 2:
		glb.datasetFolder = datasetFolders[1]
	else:
		print("Invalid input")
	print("Now working on the dataset in: " + glb.datasetFolder)

def printDatasetFolder():
	print("Now working on the dataset in: " + glb.datasetFolder)

def shapeArray(length, direction=0, checkTimestamp=True):
	returnlist = [[],[],[],[],[],[],[],[]]

	#global glb.mutex
	#global printlock
	if checkTimestamp:
		delayconstant = 0
		if direction == 5:
			delayconstant = 0.2
		else:
			if length == shortLength:
				delayconstant = 0.8
			elif length == longLength:
				delayconstant = 0.6

		startTime = tme.time() + delayconstant
		ready = False

		temp = None
		#print("Entering busy wait")
		while not ready:
			with glb.mutex:
				if len(glb.data[0][timestamp]) < (length):
					print("Lenght of data is too small")
					print(len(glb.data[0][timestamp]))
					print(length)
					return -1
				elif glb.data[0][timestamp][-1] >= startTime: #Gaar out of range
					ready = True
					temp = copy.deepcopy(glb.data)
					error = False
					for i in range(numCh):

						if temp[i][timestamp][-1] != temp[i-1][timestamp][-1]:
							print("Timestamp error, unequal for same index")
							return -1
						if len(temp[i][rawdata]) != len(temp[i-1][rawdata]):
							print("Uneven length of rawdata")
							return -1
						if len(temp[i][filterdata]) != len(temp[i-1][filterdata]):
							print("Uneven length of filterdata")
							return -1
						if len(temp[i][timestamp]) != len(temp[i-1][timestamp]):
							print("Uneven length of timestamp")
							return -1
						if len(temp[i][timestamp]) != len(temp[i][rawdata]):
							print("Uneven length between timestamp and rawdata")
							return -1

		#print("Exiting busy wait")
		stopindex = len(temp[0][rawdata])-1


		for i in range((len(temp[0][timestamp])-1), 0, -1):
			#print(i)
			if temp[0][timestamp][i]<=startTime:
				stopindex = i
				break
			elif i == (length):
				print("Index error, could not find stop index")
				return -1

		stop = stopindex
		start = stop - length
		#print("Found indexes and starting to append")
	else:
		if len(glb.data[0][timestamp]) < (length):
			print("Lenght of data is too small")
			return -1
		else:
			stop = -1
			start = -(length+1)
			temp = glb.data
	for i in range(numCh):
		returnlist[i].append(temp[i][filterdata][start:stop])
	return returnlist


def findPeakIndex(data, plot=False, label=""):
	fp = frontPadding
	golayData = signal.savgol_filter(x=data, window_length=35, polyorder = 2)
	
	if (len(data) - (frontPadding + backPadding)) < 250:
		fp = frontPadding - (250 - (len(data) - (frontPadding + backPadding)))
	
	originalWindow = golayData[fp:-backPadding]

	maxPeakIndex = np.argmax(originalWindow)
	minPeakIndex = np.argmin(originalWindow)

	if abs(originalWindow[maxPeakIndex]) > abs(originalWindow[minPeakIndex]):
		peakIndex = maxPeakIndex
	else: 
		peakIndex = minPeakIndex
	peakIndex = peakIndex - (frontPadding-fp)
	if plot:
		if len(label)>2:
			movement = MergeDict[int(label[-1])]
			movementString = movements[movement]
			movementString = movementString.replace('_',' ')
			
			plt.suptitle(movementString[1:])
			
		peakPlot = np.zeros(len(originalWindow))
		peakPlot[peakIndex] = 100
		legends = []
		legend, = plt.plot(data[fp:-backPadding], label="Original data x[n]")
		legends.append(legend)
		legend, = plt.plot(golayData[fp:-backPadding], label="Savitzky-Golay filter")
		legends.append(legend)
		legend, = plt.plot(peakPlot, label="Peakfinder")
		legends.append(legend)

		plt.ylabel('uV')
		plt.xlabel('Sample')
		plt.legend(handles=legends)
		
		plt.show()
	#print(peakIndex)
	return peakIndex

def rmPadding(data, peakIndex, windowLength=250):
	#print(len(data))
	if windowLength == None:
		windowLength = shortLength
	centerIndex = frontPadding + peakIndex

	halfWindow = windowLength // 2
	startIndex = centerIndex - halfWindow
	stopIndex = centerIndex + halfWindow
	#if len(windowLength) % 2 == 0: #window is even length

	#else: #window is odd length
	#print("Start Index: %d " %startIndex)
	#print("Stop Index: %d " %stopIndex)			

	data = data[startIndex:stopIndex]
	#print("Length of Data: %d" %len(data))
	return data

def mergeDatasets(x1, x2, y1, y2): #Merge set 2 into set 1
	#print("Len x2")
	#print(len(x2))
	#print("len x2[0]")
	#print(len(x2[0]))
	
	#print("Len y2")
	#print(len(y2))
	
	if isinstance(y2[0], int):
		y1 = np.append(y1, y2, axis = 0)
		x1 = np.append(x1, x2, axis = 0)

		'''
		for j in range(len(x2)):
			print("j = %d" %j)
			for k in range(len(x2[j])):
				print("k = %d" %k)
				x1[j].append(x2[j][k])
			y1.append(y2[j])
		'''
	else:
		#print("len y2[0]")
		#print(len(y2[0]))	
		for j in range(len(x2)):
			for k in range(len(x2[j])):
				x1[j].append(x2[j][k])
				y1[j].append(y2[j][k])

	return x1, y1
	