import os, shutil
import threading
#from filterlib import *
from globalvar import *
from operator import sub
import plot
import matplotlib.pyplot as plt
import filterlib

#TODO Lag fft export av dataset
#TODO Lag machinelearning load av dataset 



filelock = Lock()
numCh = 8
longLength = 500
shortLength = 250
frontPadding = 250
backPadding = 25
def saveShortTemp(direction):
	global shortLength, filelock
	f = prepSaveData(direction, shortLength)
	if f != -1:
		with filelock:
			file = open('Dataset\\temp.txt', 'a')
			file.write(f)
			file.close()
		#print("Save short temp completed")
	else:
		print("Failed to save short temp")

def saveLongTemp(direction):
	global longLength, filelock	
	f = prepSaveData(direction, longLength)
	if f != -1:
		with filelock:
			file = open('Dataset\\longtemp.txt', 'a')
			file.write(f)
			file.close()
		#print("Save long temp completed")
	else:
		print("Failed to save long temp")

def prepSaveData(direction, length):
	global data, nSamples
	global rawdata, filterdata, timestamp
	global timeData
	global mutex, filelock
	global numCh
	global frontPadding, backPadding
	
	if direction != 5:
		startTime = tme.time() + 0.6
	else: 
		startTime = tme.time()
	
	ready = False

	while not ready:
		if data[1][timestamp][-1] > startTime:
			ready = True

	with(mutex):
		temp = data

	if len(temp[1][rawdata]) > (length + frontPadding + backPadding):

		stopindex = len(temp[1][rawdata])-5 + backPadding

		for i in range(len(temp[timestamp])-1, 0, -1):
			if temp[1][timestamp][i]<=startTime:
				stopindex = i
				break

		stop = stopindex
		start = stop - (length + frontPadding)
		if stop > len(temp[0][timestamp])-1:
			print("Index error, aborting save operation")
		else: 
			
			abort = False
			f = ""
			for j in range(numCh):
				good = True
				if direction == 4:
					f += "lr"
				elif direction == 1:
					f +="ld"
				elif direction == 6:
					f += "rr"
				elif direction == 9:
					f += "rd"
				elif direction == 8:
					f += "ur"
				elif direction == 7:
					f += "ud"
				elif direction == 2:
					f += "dr"
				elif direction == 3:
					f += "dd"
				elif direction == 5:
					f += "cs"
				elif direction == 0:
					f += "cb"
				else:
					good = False

				if abort:
					break

				if good:
					f += str(j)
					for i in range(start, stop):
						f += ","
						try:
							num = temp[j][rawdata][i]
						except IndexError:
							print("Index error")
							print("Len buffer = %d" % len(temp[j][rawdata])-1)
							print("Index = %d" % i)
							print("Write operation aborted")
							abort = True
							break

						f += str(num)

					f += ":"

			if not abort:
				return f
			else:
				return -1

					



def exportPlots(command):
	global numCh, frontPadding, backPadding
	#PATH er feil
	dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	#folder = dir_path + "\\Dataset_exports\\figures\\Center"
	#print(folder)
	if command == "temp":
		folders = ["\\tempfigures", "\\longtempfigures"]

	elif command == "data":
		folders == ["\\figures", "\\longfigures"]
	else:
		folders = ["\\figures", "\\tempfigures", "\\longfigures", "\\longtempfigures"]
	
	movements = ["\\Center_blink", "\\Center_still", 
				"\\Down_direction", "\\Down_return", 
				"\\Up_direction", "\\Up_return",
				"\\Left_direction", "\\Left_return", 
				"\\Right_direction", "\\Right_return", "\\Garbage"]
	channels = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]


	for i in range(len(folders)):
		print("i = %d" %i)
		for j in range(len(movements)):
			folder = dir_path + "\\Dataset_exports" + folders[i] + movements[j]
			print(folder)

			for the_file in os.listdir(folder):
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
				file = open('Dataset\\temp.txt', 'r')
			else:
				file = open('Dataset\\longtemp.txt', 'r')
		elif command == "data":
			if i == 0:
				file = open('Dataset\\data.txt', 'r')
			else:
				file = open('Dataset\\longdata.txt', 'r')
		else:	

			if i == 0:
				file = open('Dataset\\data.txt', 'r')
			elif i == 1: 
				file = open('Dataset\\temp.txt', 'r')
			elif i == 2:
				file = open('Dataset\\longdata.txt', 'r')
			elif i == 3:
				file = open('Dataset\\longtemp.txt', 'r')
		

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
			#print("Feature: \n")
			#print(feature)
			#print(len(feature))
			if len(feature) > 10:
				featuretype = feature[0]
				feature.pop(0)
				#print("Feature: \n")
				#print(feature)
				print("Featuretype: \n")
				print(featuretype)

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
					print(title)
					#print(featuretype1)
					b, a = filterlib.designplotfilter()
					for l in range(0, numCh):
						#print("l = %d" %l)
						feature1 = DataSet[k+l].split(',')
						featuretype1 = feature1[0]
						feature1.pop(0)

						if not(featuretype[0] == featuretype1[0]):
							break

						featureData1 = map(float, feature1)
						subplotnum = (numCh/2)*100 + 20 + l + 1
						#print("Subplotnum = %d" %subplotnum)
						ax1 = plt.subplot(subplotnum)
						featureData1 = filterlib.plotfilter(featureData1, b, a)
						featureData1 = featureData1[frontPadding:-backPadding] #Remove paddings
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

					
					savestring = dir_path + "\\Dataset_exports\\"+tempOrNot +"\\"+title +"\\"+ title+str(k/numCh) + ".png"
					print(savestring)

					plt.subplots_adjust(hspace=0.2)
					with filelock:
						plt.savefig(savestring, bbox_inches='tight')
					
					#plt.show()
					plt.close()
			else:
				print("Empty file")
	#plt.close('all')
	print("Finished exporting plots")




def deleteShortDataelement(index):
	global length, numCh
	index = index * numCh
	file = open('Dataset\\data.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in numCh:
		DataSet.pop(index)
	#DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open('Dataset\\data.txt', 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / numCh
	print("Data element %d is deleted" % index)

def deleteLongDataelement(index):
	global length, numCh
	index = index * numCh
	file = open('Dataset\\longdata.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in numCh:
		DataSet.pop(index)
	#DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open('Dataset\\longdata.txt', 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / numCh
	print("Data element %d is deleted" % index)

def deleteShortTempelement(index):
	global length, numCh
	index = index * numCh
	file = open('Dataset\\temp.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in numCh:
		DataSet.pop(index)
	#DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open('Dataset\\temp.txt', 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / numCh
	print("Temp element %d is deleted" % index)

def deleteLongTempelement(index):
	global length, numCh
	index = index * numCh
	file = open('Dataset\\longtemp.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	for i in numCh:
		DataSet.pop(index)
	#DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open('Dataset\\longtemp.txt', 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / numCh
	print("Temp element %d is deleted" % index)

def saveShortData():
	tempfile = open('Dataset\\temp.txt', 'r')
	tempData = tempfile.read()
	tempfile.close()
	permfile = open('Dataset\\data.txt', 'a')
	permfile.write(tempData)
	permfile.close()
	tempfile = open('Dataset\\temp.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Short Data Saved")

def saveLongData():
	tempfile = open('Dataset\\longtemp.txt', 'r')
	tempData = tempfile.read()
	tempfile.close()
	permfile = open('Dataset\\longdata.txt', 'a')
	permfile.write(tempData)
	permfile.close()
	tempfile = open('Dataset\\longtemp.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Long Data Saved")


def clearShortTemp():
	tempfile = open('Dataset\\temp.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Short Temp is cleared")

def clearLongTemp():
	tempfile = open('Dataset\\longtemp.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Long Temp is cleared")


def clearShortData():
	tempfile = open('Dataset\\data.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Short Data is deleted")
	
def clearLongData():
	tempfile = open('Dataset\\longdata.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Long Data is deleted")



