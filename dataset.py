import os, shutil
import threading
#from filterlib import *
from globalvar import *
filelock = Lock()
from operator import sub
import plot
import matplotlib.pyplot as plt
import filterlib

#TODO
#fil path er feil i alle funksjoner
numCh = 8

def saveShortTemp(direction):
	global shortLength
	f = prepSaveData(direction, shortLength)
	file = open('Dataset\\temp.txt', 'a')
	file.write(f)
	file.close()

def saveLongtemp(direction):
	global longLength	
	f = prepSaveData(direction, longLength)
	file = open('Dataset\\longtemp.txt', 'a')
	file.write(f)
	file.close()

def prepSaveData(direction, length):
	global data, nSamples
	global rawdata, filterdata, timestamp
	global timeData
	global mutex
	global numCh
	
	
	if direction != 5:
		startTime = tme.time() + 0.4
	else: 
		startTime = tme.time()
	
	ready = False

	while not ready:
		if data[1][timestamp][-1] > startTime:
			ready = True

	with(mutex):
		temp = data

	if len(temp[1][rawdata]) > length:

		stopindex = len(temp[1][rawdata])-5

		for i in range(len(temp[timestamp])-1, 0, -1):
			if temp[1][timestamp][i]<=startTime:
				stopindex = i
				break

		stop = stopindex
		start = stop - length
		if stop > len(temp[0][timestamp])-1:
			print("Index error, aborting save operation")
		else: 
			with(filelock):

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

					



def exportPlots():
	global numCh
	#PATH er feil
	dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	#folder = dir_path + "\\Dataset_exports\\figures\\Center"
	#print(folder)
	folders = ["\\figures", "\\tempfigures", "\\longfigures", "\\longtempfigures"]
	movements = ["\\Center_blink", "\\Center_still", 
				"\\Down_direction", "\\Down_return", 
				"\\Up_direction", "\\Up_return",
				"\\Left_direction", "\\Left_return", 
				"\\Right_direction", "\\Right_return", "\\Garbage"]
	channels = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]


	for i in range(len(folders)):
		print(len(folders))
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
			care = True
			feature = []
			feature = DataSet[k].split(',')
			featuretype = feature[0]
			feature.pop(0)
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
				plt.suptitle(title)
				#print(title)
				#print(featuretype1)
				b, a = filterlib.designplotfilter()
				for l in range(0, numCh):
					feature1 = DataSet[k+l].split(',')
					featuretype1 = feature1[0]
					feature1.pop(0)

					if not(featuretype[0] == featuretype1[0]):
						break

					featureData1 = map(float, feature1)
					subplotnum = (numCh/2)*100 + 20 + j + 1
					ax1 = plt.subplot(subplotnum)
					featureData1 = filterlib.plotfilter(featureData1, b, a)
					plot.exportplot(featureData1, channels[l], ax1)


				tempOrNot = None
				if i == 0:
					tempOrNot = "figures"
				elif i == 1:
					tempOrNot = "tempfigures"
				elif i == 2:
					tempOrNot = "longfigures"
				elif i == 3:
					tempOrNot = "longtempfigures"


				savestring = dir_path + "\\Dataset_exports\\"+tempOrNot +"\\"+title +"\\"+ title+str(i/numCh) + ".png"
				print(savestring)
				plt.subplots_adjust(hspace=0.45)
				with filelock:
					plt.savefig(savestring, bbox_inches='tight')
				
				#plt.show()
				plt.close()
	#plt.close('all')
	print("Finished exporting plots")



def viewdataelement(index): #Utdatert
	global length
	file = open('Dataset\\data.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	i = index * 2
	if i > len(DataSet)-2:
		print("Value is too big")
	else:
		care = True
		feature1 = []
		feature1 = DataSet[i].split(',')
		featuretype1 = feature1[0]
		feature1.pop(0)
		if featuretype1 == 'u0':
			title = "Up"
		elif featuretype1 == 'd0':
			title = "Down"
		elif featuretype1 == 'l0':
			title = "Left"
		elif featuretype1 == 'r0':
			title = "Right"
		elif featuretype1 == 'c0':
			title = "Center"
			#care = False
		else:
			title = featuretype1
			care = False

		if care:
			plt.suptitle(title)
			print(title)
			print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-100,100])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("Fp1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-100,100])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("Fp2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			savestring = "Dataset_exports/figures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			#plt.savefig(savestring, bbox_inches='tight')
			plt.show()
			#plt.close()

def viewtempelement(index): #Utdatert
	global length
	file = open('Dataset\\temp.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	i = index * 2
	if i > len(DataSet)-2:
		print("Value is too big")
	else:
		care = True
		feature1 = []
		feature1 = DataSet[i].split(',')
		featuretype1 = feature1[0]
		feature1.pop(0)
		if featuretype1 == 'u0':
			title = "Up"
		elif featuretype1 == 'd0':
			title = "Down"
		elif featuretype1 == 'l0':
			title = "Left"
		elif featuretype1 == 'r0':
			title = "Right"
		elif featuretype1 == 'c0':
			title = "Center"
			#care = False
		else:
			title = featuretype1
			care = False

		if care:
			plt.suptitle(title)
			print(title)
			print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-100,100])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("Fp1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-100,100])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("Fp2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			savestring = "figures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			#plt.savefig(savestring, bbox_inches='tight')
			plt.show()
			#plt.close()
def deletedataelement(index):
	global length, numCh
	index = index * 2
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
	index = index / 2
	print("Data element %d is deleted" % index)

def deletelongdataelement(index):
	global length, numCh
	index = index * 2
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
	index = index / 2
	print("Data element %d is deleted" % index)

def deletetempelement(index):
	global length, numCh
	index = index * 2
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
	index = index / 2
	print("Temp element %d is deleted" % index)

def deletelongtempelement(index):
	global length, numCh
	index = index * 2
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
	index = index / 2
	print("Temp element %d is deleted" % index)

def saveData():
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

def clearTemp():
	tempfile = open('Dataset\\temp.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Short Temp is cleared")

def clearLongTemp():
	tempfile = open('Dataset\\temp.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Long Temp is cleared")

def clearData():
	tempfile = open('Dataset\\data.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Short Data is deleted")

def clearLongData():
	tempfile = open('Dataset\\data.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Long Data is deleted")

def databufferPop():
	global data
	with mutex:
		data[i][rawdata].pop(0)
		data[i][filterdata].pop(0)
		data[i][timestamp].pop(0)