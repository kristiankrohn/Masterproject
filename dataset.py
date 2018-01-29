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


printlock = Lock()
filelock = Lock()

longLength = 2.5 * glb.fs
shortLength = glb.fs

frontPadding = 3 * glb.fs
backPadding = glb.fs


def saveShortTemp(direction):
	global filelock, printlock
	f = prepSaveData(direction, shortLength)

	if f != -1:
		with filelock:
			file = open(dir_path+"\\Dataset\\temp.txt", 'a')
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
			file = open(dir_path+"\\Dataset\\longtemp.txt", 'a')
			file.write(f)
			file.close()
		#print("Save long temp completed")
	else:
		print("Failed to save long temp")

def prepSaveData(direction, length):
	global mutex, filelock
	#global printlock
	if length == shortLength:
		delayconstant = 0.8
	elif lenght == longLength:
		delayconstant = 0.6
	startTime = tme.time() + (backPadding/glb.fs) + delayconstant
	ready = False
	if len(glb.data[0][timestamp]) < (length + frontPadding + backPadding):
		print("Lenght of data is too small")
		return -1
	temp = None
	

	while not ready:
		with mutex:

			if glb.data[0][timestamp][-1] >= startTime:
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


def exportPlots(command, plottype="time"):
	global filelock

	
	#folder = dir_path + "\\Dataset_exports\\figures\\Center"
	#print(folder)
	if command == "temp":
		folders = ["\\tempfigures", "\\longtempfigures"]

	elif command == "data":
		folders == ["\\figures", "\\longfigures"]

	else:
		folders = ["\\figures", "\\tempfigures", 
				"\\longfigures", "\\longtempfigures"]
	
	movements = ["\\Center_blink", "\\Center_still", 
				"\\Down_direction", "\\Down_return", 
				"\\Up_direction", "\\Up_return",
				"\\Left_direction", "\\Left_return", 
				"\\Right_direction", "\\Right_return", "\\Garbage"]

	channels = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]


	for i in range(len(folders)):
	
		for j in range(len(movements)):
			if plottype == "fft":
				folder = dir_path + "\\Dataset_fft" + folders[i] + movements[j]
			elif plottype == "time":
				folder = dir_path + "\\Dataset_exports" + folders[i] + movements[j]
			elif plottype == "raw":
				folder = dir_path + "\\Dataset_raw" + folders[i] + movements[j]
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
				file = open(dir_path+"\\Dataset\\temp.txt", 'r')
			else:
				file = open(dir_path+"\\Dataset\\longtemp.txt", 'r')
		elif command == "data":
			if i == 0:
				file = open(dir_path+"\\Dataset\\data.txt", 'r')
			else:
				file = open(dir_path+"\\Dataset\\longdata.txt", 'r')
		else:	

			if i == 0:
				file = open(dir_path+"\\Dataset\\data.txt", 'r')
			elif i == 1: 
				file = open(dir_path+"\\Dataset\\temp.txt", 'r')
			elif i == 2:
				file = open(dir_path+"\\Dataset\\longdata.txt", 'r')
			elif i == 3:
				file = open(dir_path+"\\Dataset\\longtemp.txt", 'r')
		

		with filelock:	
			AllData = file.read()
			file.close()
		
		DataSet = []
		DataSet = AllData.split(':')
		#Definer numch skikkelig!!
		for k in range(0, len(DataSet), numCh):
			#print("k = %d" %k)
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
							#featureData1 = featureData1[frontPadding:-backPadding] #Remove paddings
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
						savestring = (dir_path + "\\Dataset_fft\\"
										+ tempOrNot +"\\"+title +"\\"
										+ title+str(k/numCh) + ".png")
					elif plottype == "raw":
						savestring = (dir_path + "\\Dataset_raw\\"
										+ tempOrNot +"\\"+title +"\\"
										+ title+str(k/numCh) + ".png")
					else:
						savestring = (dir_path + "\\Dataset_exports\\"
										+ tempOrNot +"\\"+title +"\\"
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




def deleteShortDataelement(index):

	index = index * numCh
	file = open(dir_path+"\\Dataset\\data.txt", 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in numCh:
		DataSet.pop(index)
	file.close()
	file = open(dir_path+"\\Dataset\\data.txt", 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / numCh
	print("Data element %d is deleted" % index)

def deleteLongDataelement(index):

	index = index * numCh
	file = open(dir_path+"\\Dataset\\longdata.txt", 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in numCh:
		DataSet.pop(index)
	file.close()
	file = open(dir_path+"\\Dataset\\longdata.txt", 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / numCh
	print("Data element %d is deleted" % index)

def deleteShortTempelement(index):

	index = index * numCh
	file = open(dir_path+"\\Dataset\\temp.txt", 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in range(numCh):
		DataSet.pop(index)
	#DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open(dir_path+"\\Dataset\\temp.txt", 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / numCh
	print("Temp element %d is deleted" % index)

def deleteLongTempelement(index):

	index = index * numCh
	file = open(dir_path+"\\Dataset\\longtemp.txt", 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in range(numCh):
		DataSet.pop(index)
	#DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open(dir_path+"\\Dataset\\longtemp.txt", 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / numCh
	print("Temp element %d is deleted" % index)

def saveShortData():

	tempfile = open(dir_path+"\\Dataset\\temp.txt", 'r')
	tempData = tempfile.read()
	tempfile.close()
	permfile = open(dir_path+"\\Dataset\\data.txt", 'a')
	permfile.write(tempData)
	permfile.close()
	tempfile = open(dir_path+"\\Dataset\\temp.txt", 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Short Data Saved")

def saveLongData():

	tempfile = open(dir_path+"\\Dataset\\longtemp.txt", 'r')
	tempData = tempfile.read()
	tempfile.close()
	permfile = open(dir_path+"\\Dataset\\longdata.txt", 'a')
	permfile.write(tempData)
	permfile.close()
	tempfile = open(dir_path+"\\Dataset\\longtemp.txt", 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Long Data Saved")


def clearShortTemp():

	tempfile = open(dir_path+"\\Dataset\\temp.txt", 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Short Temp is cleared")

def clearLongTemp():

	tempfile = open(dir_path+"\\Dataset\\longtemp.txt", 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Long Temp is cleared")


def clearShortData():

	tempfile = open(dir_path+"\\Dataset\\data.txt", 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Short Data is deleted")
	
def clearLongData():

	tempfile = open(dir_path+"\\Dataset\\longdata.txt", 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Long Data is deleted")

def loadDataset(filename="data.txt"):
	global filelock
	print("Starting to load dataset")
	x = [[],[],[],[],[],[],[],[]]
	y = [[],[],[],[],[],[],[],[]]

	file = None

	with filelock:
		file = open((dir_path+"\\Dataset\\"+filename), 'r')
		AllData = file.read()
		file.close()
		
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	b, a = filterlib.designfilter()
	for k in range(0, len(DataSet)):
		#print("k = %d" %k)
		care = True
		feature = []
		feature = DataSet[k].split(',')
		#print("Length of vector: %d\n" %len(feature))
		if len(feature) > 10:
			featuretype = feature[0]
			label = featuretype[0:2]
			channel = map(int, featuretype[2])
			channel = channel[0]
			feature.pop(0)
			
			y[channel].append(directioncode.index(label))							
			
			featureData = map(float, feature)
			#print("Raw data: %0.2f\n" %featureData[0])
			featureData = filterlib.plotfilter(featureData, b, a)
			featureData = featureData[frontPadding:-backPadding] #Remove paddings
			#print("Filterdata: %0.2f\n" %featureData[0])
			x[channel].append(featureData)
		#else:
			#print("Invalid datapoint")
			#print(feature)
	#plt.close('all')
	print("Finished loading dataset")
	return(x,y)


