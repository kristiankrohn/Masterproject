import os, shutil
import threading
#from filterlib import *
from globalvar import *
filelock = Lock()
from operator import sub

#TODO
#fil path er feil i alle funksjoner


def saveTempData(direction):
	global data, nSamples
	global timeData
	global mutex
	global length
	numCh = 2
	
	if direction != 5:
		startTime = tme.time() + 0.4
	else: 
		startTime = tme.time()
	
	ready = False

	while not ready:
		if timeData[1][-1] > startTime:
			ready = True

	with(mutex):
		temp = data
		tempTime = timeData

	if len(temp[1]) > length:

		stopindex = len(temp[1])-5

		for i in range(len(tempTime[1])-1, 0, -1):
			if tempTime[1][i]<=startTime:
				stopindex = i
				break

		stop = stopindex
		start = stop - length
		if stop > len(temp[0])-1:
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
						f += "cr"
					else:
						good = False

					if abort:
						break

					if good:
						f += str(j)
						for i in range(start, stop):
							f += ","
							try:
								num = temp[j][i]
							except IndexError:
								print("Index error")
								print("Len buffer = %d" % len(temp[j])-1)
								print("Index = %d" % i)
								print("Write operation aborted")
								abort = True
								break

							f += str(num)

						f += ":"

				if not abort:		
					file = open('Dataset\\temp.txt', 'a')
					file.write(f)
					file.close()



def openFile():
	global length
	#PATH er feil
	dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	folder = dir_path + "\\Dataset_exports\\figures\\Center"
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
	folder = dir_path + "\\Dataset_exports\\figures\\Down"
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
	folder = dir_path + "\\Dataset_exports\\figures\\Left"
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
	folder = dir_path + "\\Dataset_exports\\figures\\Right"
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
	folder = dir_path + "\\Dataset_exports\\figures\\Up"
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

	file = open('Dataset\\data.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	for i in range(0, len(DataSet), 4):
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
			#print(title)
			#print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-100,100])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("X1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			#print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-100,100])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("X2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')


			#featureData3 = map(sub, featureData1, featureData2)
			#ax3 = plt.subplot(313)
			#ax3.set_autoscaley_on(False)
			#ax3.set_ylim([-100,100])
			#ax3.set_title("Fp1 - Fp2")
			#plt.plot(x, featureData3, label='Fp1 - Fp2')
			#plt.ylabel('uV')
			#plt.xlabel('Seconds')

			#featureData3 = map(sub, featureData1, featureData2)
			#ax3 = plt.subplot(313)
			#ax3.set_autoscaley_on(False)
			#ax3.set_xlim([1,20])
			#ax3.set_title("Fp1 - Fp2")
			#Y = abs(np.fft.fft(featureData1)/len(featureData1))
			
			#plt.plot(Y, label='Fp1 - Fp2')
			#plt.ylabel('uV')
			#plt.xlabel('Hz')
			#print(signal.hilbert(featureData1))
			savestring = "figures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			plt.savefig(savestring, bbox_inches='tight')
			#plt.show()
			plt.close()
	#plt.close('all')
	print("Finished exporting plots")

def opentemp():
	global length

	dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	folder = dir_path + "\\Dataset_exports\\tempfigures\\Center"
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
	folder = dir_path + "\\Dataset_exports\\tempfigures\\Down"
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
	folder = dir_path + "\\Dataset_exports\\tempfigures\\Left"
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
	folder = dir_path + "\\Dataset_exports\\tempfigures\\Right"
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
	folder = dir_path + "\\Dataset_exports\\tempfigures\\Up"
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

	file = open('Dataset\\temp.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	for i in range(0, len(DataSet), 2):
		care = True
		feature1 = []
		feature1 = DataSet[i].split(',')
		featuretype1 = feature1[0]
		feature1.pop(0)
		if featuretype1 == 'ur0':
			title = "Up"
		elif featuretype1 == 'dr0':
			title = "Down"
		elif featuretype1 == 'lr0':
			title = "Left"
		elif featuretype1 == 'rr0':
			title = "Right"
		elif featuretype1 == 'cr0':
			title = "Center"
			#care = False
		else:
			title = featuretype1
			care = False

		if care:
			plt.suptitle(title)
			#print(title)
			#print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-200,200])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("X1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			#print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-200,200])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("X2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			savestring = "tempfigures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			plt.savefig(savestring, bbox_inches='tight')
			#plt.show()
			plt.close()
	plt.close('all')
	print("Finished exporting plots")
def viewdataelement(index):
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

			savestring = "figures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			#plt.savefig(savestring, bbox_inches='tight')
			plt.show()
			#plt.close()

def viewtempelement(index):
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
	global length
	index = index * 2
	file = open('Dataset\\data.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	DataSet.pop(index)
	DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open('Dataset\\data.txt', 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / 2
	print("Data element %d is deleted" % index)

def deletetempelement(index):
	global length
	index = index * 2
	file = open('Dataset\\temp.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	DataSet.pop(index)
	DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open('Dataset\\temp.txt', 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / 2
	print("Temp element %d is deleted" % index)

def parametrization(map1, map2):
	map3 = map1 - map2


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
	print("Saved")

def clearTemp():
	tempfile = open('Dataset\\temp.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Temp is cleared")

def clearData():
	tempfile = open('Dataset\\data.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Data is deleted")