import os, shutil
import threading
#from filterlib import *
from globalvar import *
from operator import sub
import plot
import matplotlib.pyplot as plt
import filterlib
import time as tme
import copy


printlock = Lock()
filelock = Lock()

longLength = 500
shortLength = 250
frontPadding = 750
backPadding = 125


def saveShortTemp(direction):

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
	#global data, nSamples
	#global rawdata, filterdata, timestamp
	#global timeData
	global mutex, filelock
	#global numCh
	#global frontPadding, backPadding
	#global directioncode
	global printlock
	
	startTime = tme.time() + (backPadding/fs)
	ready = False
	if len(data[0][timestamp]) < (length + frontPadding + backPadding):
		print("Lenght of data is too small")
		return -1
	temp = None
	temptemp = None
	with printlock:
		while not ready:
			with mutex:

				if data[0][timestamp][-1] >= startTime:
					ready = True
					temp = copy.deepcopy(data)
					error = False
					for i in range(numCh):
						#print(temp[i][timestamp][-1])

						print("Checking for errors")

						if temp[i][timestamp][-1] != temp[i-1][timestamp][-1]:
							ready = False
							error = True
						if len(temp[i][rawdata]) != len(temp[i-1][rawdata]):
							print("Uneven length of rawdata")
							error = True
						if len(temp[i][filterdata]) != len(temp[i-1][filterdata]):
							print("Uneven length of filterdata")
							error = True
						if len(temp[i][timestamp]) != len(temp[i-1][timestamp]):
							print("Uneven length of timestamp")
							error = True
						if len(temp[i][timestamp]) != len(temp[i][rawdata]):
							error = True
						if error:
							return -1	

		stopindex = len(temp[0][rawdata])-1


		for i in range((len(temp[0][timestamp])-1), (length + frontPadding + backPadding), -1): #Dette her fungerer ikke
			#print(i)
			if temp[0][timestamp][i]<=startTime:
				stopindex = i
				print("Found Stop index at: %d" %i)
				print(startTime)
				print(temp[0][timestamp][i])
				break
			elif i == (length + frontPadding + backPadding):
				print("Index error, could not find stop index")
				return -1

		stop = stopindex
		start = stop - (length + frontPadding + backPadding)
		print("Start index: %d" %start)
		print("Stop index: %d" %stop)
			
		abort = False
		f = ""
		for j in range(numCh):
			
			good = True

			if direction < 10:
				f += directioncode[direction]	
			else:
				good = False

			if good:
				f += str(j) #Kanal
				
				'''
				print("\nLast elements in array:")
				print(temp[j][rawdata][(stop-10):])
				print("index start:")
				print(start)
				print("index stop:")
				print(stop)
				print("Length of array")
				print(len(temp[j][rawdata]))
				'''
				for i in range(start, stop):
					f += ","
					try:
						num = temp[j][rawdata][i]

					except IndexError:
						print("Index error")
						print("Len buffer = %d" %(len(temp[j][rawdata])-1))
						print("Index = %d" % i)
						print("Write operation aborted")
						abort = True
						return -1
					'''
					if i == (stop-10):
						print("Last ten elements in savestring")
					elif i >= (stop - 10):
						print(str(num)) 
					'''
					f += str(num)

				f += ":"
				
		if not abort:

			return f
		else:
			print("Aborted save operation")
			return -1


						



def exportPlots(command, plottype="time"):
	global numCh, frontPadding, backPadding, filelock, dir_path

	
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
			print("k = %d" %k)
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

					b, a = filterlib.designplotfilter()
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
							featureData1 = featureData1[frontPadding:-backPadding] #Remove paddings
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
	global length, numCh, dir_path
	index = index * numCh
	file = open(dir_path+"\\Dataset\\data.txt", 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in numCh:
		DataSet.pop(index)
	#DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open(dir_path+"\\Dataset\\data.txt", 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / numCh
	print("Data element %d is deleted" % index)

def deleteLongDataelement(index):
	global length, numCh, dir_path
	index = index * numCh
	file = open(dir_path+"\\Dataset\\longdata.txt", 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in numCh:
		DataSet.pop(index)
	#DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open(dir_path+"\\Dataset\\longdata.txt", 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / numCh
	print("Data element %d is deleted" % index)

def deleteShortTempelement(index):
	global length, numCh, dir_path
	index = index * numCh
	file = open(dir_path+"\\Dataset\\temp.txt", 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in numCh:
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
	global length, numCh, dir_path
	index = index * numCh
	file = open(dir_path+"\\Dataset\\longtemp.txt", 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in numCh:
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
	global dir_path
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
	global dir_path
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
	global dir_path
	tempfile = open(dir_path+"\\Dataset\\temp.txt", 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Short Temp is cleared")

def clearLongTemp():
	global dir_path
	tempfile = open(dir_path+"\\Dataset\\longtemp.txt", 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Long Temp is cleared")


def clearShortData():
	global dir_path
	tempfile = open(dir_path+"\\Dataset\\data.txt", 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Short Data is deleted")
	
def clearLongData():
	global dir_path
	tempfile = open(dir_path+"\\Dataset\\longdata.txt", 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Long Data is deleted")

def loadDataset(filename="data.txt"):
	global numCh, frontPadding, backPadding, filelock, dir_path, directioncode
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
	b, a = filterlib.designplotfilter()
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
			#featureData = filterlib.plotfilter(featureData, b, a)
			#featureData = featureData[frontPadding:-backPadding] #Remove paddings
			#print("Filterdata: %0.2f\n" %featureData[0])
			x[channel].append(featureData)
		#else:
			#print("Invalid datapoint")
			#print(feature)
	#plt.close('all')
	print("Finished loading dataset")
	return(x,y)


