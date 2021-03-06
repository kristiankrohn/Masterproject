import sys; sys.path.append('.') # help python find open_bci_v3.py relative to scripts folder
sys.path.append('../OpenBCI')
import open_bci_v3 as bci
import os
import logging
import time as tme
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.ptime import time
import threading
from scipy import signal
import scipy.fftpack
import matplotlib.pyplot as plt
from globalconst import *
import globalvar as glb
import gui as ttk
import plot as plotlib
import filterlib 
import dataset
import learning #this is moved to keys() -> "learn"
import serial
from datetime import timedelta
import hht
from datetime import datetime
import classifier
import predict
import webbrowser
import controller
import features
import tracestack
import speak
import executiontime

####TODO############################################
##
##	Test ny implementasjon av classifier
##	Sjekk Adrian dataset for lengde, 
##	short length var satt til 100 nar settet ble laget
##
####END TODO########################################

#tracestack.on(prompt=True)


board = None
root = None
graphVar = False
guiVar = False
count = None
curves = []
ptr = 0
p = None
exit = False
filtering = True
killkeys = False
predictioncondition = False
predictioninterval = 50 #Number of new samples before a new prediction, 25 with window length of 100 worked fine
predictionParameters = None
counterlock = Lock()

intervalcounter = 0
dataset.printDatasetFolder()
predictionParameters = predict.loadPredictor(machinestate)
print(predictionParameters)
executionTimeList = executiontime.readExecutionTime(filename = 'executionTimeAllFeatures200.txt')
print("Executiontime for featuremask: %.6f" %executiontime.getExecutionCost(predictionParameters['featuremask'], executionTimeList))

error = False

## for debugging dataCatcher
start = datetime.now()
oldStart = datetime.now()


def setpriority(priority=2):
	import psutil
	p = psutil.Process(os.getpid())
	print("Original priority:")
	print(p.nice())

	priorityclasses = [psutil.IDLE_PRIORITY_CLASS,
						psutil.BELOW_NORMAL_PRIORITY_CLASS,
						psutil.NORMAL_PRIORITY_CLASS,
						psutil.ABOVE_NORMAL_PRIORITY_CLASS,
						psutil.HIGH_PRIORITY_CLASS,
						psutil.REALTIME_PRIORITY_CLASS]

	p.nice(priorityclasses[priority])
	print("New priority:")
	print(p.nice())

def check_sleep(amount):
    start = datetime.now()
    tme.sleep(amount)
    end = datetime.now()
    delta = end-start
    return delta.seconds + delta.microseconds/1000000.


def housekeeper():
	global error
	longsleep = False
	pop = False
	#setpriority(5)
	start = datetime.now()
	stop = datetime.now()
	while True:
		with glb.mutex:
			if len(glb.data[0][0]) >= nSamples + 500:
				#print("Error in tme.sleep() function sleeps for too long, flushing databuffer!")
				print("Error in tme.sleep() function, sleeps for too long")
				
				for i in range(numCh):
					for j in range(3):
						del glb.data[i][j][:]
						
				error = sum(abs(check_sleep(0.050)-0.050) for i in xrange(100))*10
				print "Average error is %0.2fms" % error
				
				error = True
				break
			if len(glb.data[0][0]) >= nSamples + 100:
				longsleep = False
				pop = True
			elif len(glb.data[0][0]) >= nSamples:
				longsleep = True
				pop = True
			else:
				pop = False
			if pop == True:
				for i in range(numCh):
					glb.data[i][rawdata].pop(0)
					glb.data[i][filterdata].pop(0)
					glb.data[i][timestamp].pop(0)
				for i in range(numCh):
					if len(glb.data[i][rawdata]) != len(glb.data[i-1][rawdata]):
						print("Uneven length of rawdata")
					if len(glb.data[i][filterdata]) != len(glb.data[i-1][filterdata]):
						print("Uneven length of filterdata")
					if len(glb.data[i][timestamp]) != len(glb.data[i-1][timestamp]):
						print("Uneven length of timestamp")
					if len(glb.data[i][timestamp]) != len(glb.data[i][rawdata]):
						print("Uneven length between timestamp and raw data")

		if longsleep:
			if glb.fs == 250.0:
				start = datetime.now()
				tme.sleep(0.003)
				stop = datetime.now()
				#print(stop-start)
			else:
				tme.sleep(0.007)
		else:
			if glb.fs == 250.0:
				if (stop - start) > timedelta(milliseconds=10):
					pass
				else:
					tme.sleep(0.002)
			else:
				tme.sleep(0.004)
		
	print("Fired the housekeeper")
def dataCatcher():
	global board

	baud = 115200
	logging.basicConfig(filename="test.log",
		format='%(asctime)s - %(levelname)s : %(message)s',level=logging.DEBUG)
	logging.info('---------LOG START-------------')
	if numCh >= 9:
		for i in range(10):
			if os.name == 'nt':
				#print("Connecting to Windows serial port")

				port = "COM" + str(i)
			elif os.name == 'posix':
				#print("Connecting to Linux Serial port")
				port = "/dev/ttyS" + str(i)
			try:
				board = bci.OpenBCIBoard(port=port, scaled_output=True, log=True, 
					filter_data = False, daisy=True)
				break
			except Exception:
				pass
	else:
		for i in range(10):
			if os.name == 'nt':
				#print("Connecting to Windows serial port")
				port = "COM" + str(i)
			elif os.name == 'posix':
				#print("Connecting to Linux serial port")
				port = "/dev/ttyUSB" + str(i)
				#port = '/dev/tty.OpenBCI-DN008VTF'
				#port = '/dev/tty.OpenBCI-DN0096XA'
			try:
				board = bci.OpenBCIBoard(port=port, scaled_output=True, log=True, 
					filter_data = False, daisy=False)
				break
			except Exception:
				print("Could not connect to this port")
				pass

	if board != None:
		print("Board Instantiated")
		board.ser.write('v')
		#tme.sleep(10)

		if not board.streaming:
			board.ser.write(b'b')
			board.streaming = True

		print("Samplerate: %0.2fHz" %board.getSampleRate())
		glb.fs = board.getSampleRate()
		glb.b, glb.a = filterlib.designfilter(filtertype="notch", Q=20)
		board.start_streaming(printData)
	else:
		print("Board initialization failed, exit and reconnect dongle")


def printData(sample):	
	global intervalcounter, predictioncondition, counterlock, error
	##For debugging
	global start, oldStart
	start = datetime.now()
	#print(start-oldStart)
	#oldStart = start


	xt = tme.time()
	if glb.fs == 125.0:
		offset = 2
		start = 1
	else:
		offset = 1
		start = 0

	if ((sample.id - offset) == glb.oldSampleID) or (glb.oldSampleID == 255 and sample.id == start): 
		for i in range(numCh):			
			glb.newSamples[i].append(sample.channel_data[i])
			glb.newTimeData[i].append(xt)
		glb.oldSampleID = sample.id
	else:
		print("Lost packet, flushing databuffer")
		print("Old packet ID = %d" %glb.oldSampleID)
		print("Incomming packet ID = %d" %sample.id)
		speak.Speak("Lost packet")
		glb.oldSampleID = sample.id
		with glb.mutex:
			for i in range(numCh):
				del glb.newTimeData[i][:]
				del glb.newSamples[i][:]
				for j in range(3):
					del glb.data[i][j][:]
				
	#if predictioncondition == True:
		#print("predictioncondition is true")	
		#with counterlock:
			#print("got lock")
	intervalcounter += 1
	if intervalcounter >= predictioninterval:
		#print("New predict:")
		intervalcounter = 0

		#predict.predictRealTime(**predictionParameters)

		with glb.rtLock:
			#print("Serial rt lock")
			if not glb.rtQueue.full():            
				glb.rtQueue.put(start)
			else:
				popped = glb.rtQueue.get()
				glb.rtQueue.put(start)
			
			#predictionThread = threading.Thread(target=predict.predictRealTime,kwargs=(predictionParameters))
			#predictionThread.setDaemon(True)
			#predictionThread.start()
	

	if len(glb.newSamples[0]) >= glb.window:
		with glb.mutex:
			#sprint("From datacatcher b.glb: ")
			#print glb.b

			for i in range(numCh):
				if filterlib.filter(glb.newSamples, glb.newTimeData, i, glb.b, glb.a):
					glb.newTimeData[i][:] = []
					glb.newSamples[i][:] = []

	if error:
		with glb.mutex:
			while len(glb.data[0][0]) >= nSamples:
				for i in range(numCh):
					glb.data[i][rawdata].pop(0)
					glb.data[i][filterdata].pop(0)
					glb.data[i][timestamp].pop(0)

def update():
	global curves, p

	string = ""
	with glb.mutex:
		for i in range(nPlots):
			curves[i].setData(glb.data[i][filterdata])
			if len(glb.data[i][filterdata])>100:
				string += '   Ch: %d ' % i
				string += ' = %0.2f uV ' % glb.data[i][filterdata][-1]

	p.setTitle(string)

def graph():
	#Graph setup
	global nPlots, nSamples, count, data, curves, p, QtGui, app
	app = QtGui.QApplication([])
	p = pg.plot()

	p.setWindowTitle('pyqtgraph example: MultiPlotSpeedTest')
	#p.setRange(QtCore.QRectF(0, -10, 5000, 20)) 
	p.setLabel('bottom', 'Index', units='B')
	#curves = [p.plot(pen=(i,nPlots*1.3)) for i in range(nPlots)]

	
	
	lastTime = time()
	fps = None
	count = 0

	displayUV = []
	df = 1

	for i in range(nPlots):
		c = pg.PlotCurveItem(pen=(i,nPlots*1.3))
		p.addItem(c)
		c.setPos(0,i*100+100)
		curves.append(c)

	#p.setYRange(0, nPlots*6)
	p.setYRange(0, nPlots*100)
	p.setXRange(0, nSamples)
	p.resize(nSamples,1000)

	mw = QtGui.QMainWindow()
	mw.resize(800,800)
	
	timer = QtCore.QTimer()
	timer.timeout.connect(update)
	timer.start(0)

	print("Graphsetup finished")
	
	QtGui.QApplication.instance().exec_()


def keys():

	global board, bandstopFilter, filtering, lowpassFilter, bandpassFilter, graphVar
	global guiVar
	global predictionParameters, predictioncondition

	while not killkeys:
		inputString = raw_input()
		if "=" in inputString:
			stringArr = inputString.split('=')
			string = stringArr[0]
			if stringArr[1].isdigit():
				inputval = int(stringArr[1])
			else:
				inputval = None
		else:
			string = inputString
			inputval = None
		'''
		if string == "notch=true":
			bandstopFilter = True
		elif string == "notch=false":
			bandstopFilter = False
		elif string == "filter=true":
			filtering = True
			print(filtering)
		elif string == "filter=false":
			filtering = False
			print(filtering)
		elif string == "lowpass=true":
			lowpassFilter = True
		elif string == "lowpass=false":
			lowpassFilter = False
		elif string == "bandpass=true":
			bandpassFilter = True
		elif string == "bandpass=false":
			bandpassFilter = False
		'''
		if string == "exit":
			print("Initiating exit sequence")
			exit = True
			#if root != None:
				#root.destroy()
			#print("Quit gui")
			if QtGui != None:
				QtGui.QApplication.quit()
			#print("Quit Graph")
			if board != None:
				print("Closing board")
				board.stop()
				board.disconnect()
			#print("Quit board")
			os._exit(0)
		elif string == "plot":
			plotlib.plot()
			#plotThread = threading.Thread(target=plot,args=())
			#plotThread.start()
			#plotthread.join()
		elif string == "plotall":
			plotlib.plotAll()
			#plotAllThread = threading.Thread(target=plotAll,args=())
			#plotAllThread.start()
			#plotAllThread.join()
		#elif string == "save":
			#save()
		elif string == "start":
			threadDataCatcher = threading.Thread(target=dataCatcher,args=())
			threadDataCatcher.setDaemon(True)
			threadDataCatcher.start()

		elif string == "graph":
			#graphVar = True
			thread2 = threading.Thread(target=graph,args=())
			thread2.start()

		elif string == "gui":
			#app = ttk.App()
			guiVar = True
			#threadGui = threading.Thread(target=ttk.guiloop, args=())
			#threadGui.setDaemon(True)
			#threadGui.start()
		elif string == "makedata":
			threadDataCatcher = threading.Thread(target=dataCatcher,args=())
			#threadDataCatcher.setDaemon(True)
			threadDataCatcher.start()
			guiVar = True
			thread2 = threading.Thread(target=graph,args=())
			thread2.start()
			#graphVar = True
			#threadGui = threading.Thread(target=gui, args=())
			#threadGui.setDaemon(True)
			#threadGui.start()
		elif string == "analyzefilter":
			if inputval != None:
				filterlib.analyze_filter(inputval)
			else:
				filterlib.analyze_filter()
		elif string == "setfilter":
			if inputval != None:
				filterlib.set_filter_q(inputval)
			else:
				print("Invalid input")	

		elif string == "exportdataplots":
			exportThread = threading.Thread(target=dataset.exportPlots, 
												args=("data", "time")) 
			exportThread.start()
			#dataset.exportPlots("data")
		elif string == "exporttempplots":
			exportThread = threading.Thread(target=dataset.exportPlots, 
												args=("temp", "time","fast")) 
			exportThread.start()
			#dataset.exportPlots("temp")
		elif string == "exportallplots":
			exportThread = threading.Thread(target=dataset.exportPlots, 
												args=("all", "time")) 
			exportThread.start()
			#dataset.exportPlots("all")
		elif string == "saveshortdata":
			dataset.saveShortData()

		elif string == "savelongdata":
			dataset.saveLongData()

		elif string == "saveall":
			dataset.saveShortData()
			dataset.saveLongData()

		elif string == "deleteshortdata":
			dataset.clear("shortdata")

		elif string == "deletelongdata":
			dataset.clear("longdata")

		elif string == "deleteshorttemp":
			dataset.clear("shorttemp")

		elif string == "deletelongtemp":
			dataset.clear("longtemp")

		elif string == "deletetemp":
			dataset.clear("shorttemp")
			dataset.clear("longtemp")

		elif string == "savefilter":
			filterlib.savefiltercoeff()
			

		elif string == "deleteshortdataelement":
			if inputval != None:
				dataset.deleteelement(inputval, "data.txt")
			else:
				print("Invalid input")

		elif string == "deleteshorttempelement":
			if inputval != None:
				dataset.deleteelement(inputval, "temp.txt")
			else:
				print("Invalid input")

		elif string == "deletelongdataelement":
			if inputval != None:
				dataset.deleteelement(inputval, "longdata.txt")
			else:
				print("Invalid input")

		elif string == "deletelongtempelement":
			if inputval != None:
				dataset.deleteelement(inputval, "longtemp.txt")
			else:
				print("Invalid input")

		elif string == "deleteappendedtemp":
			dataset.deletesystem("shorttemp")
			dataset.deletesystem("longtemp")

		elif string == "deleteappendeddata":
			dataset.deletesystem("shortdata")
			dataset.deletesystem("longdata")
		
		elif string == "appenddelshorttemp":
			if inputval != None:
				dataset.appenddelete(inputval, "shorttemp")
			else:
				print("Invalid input")

		elif string == "appenddellongtemp":
			if inputval != None:
				dataset.appenddelete(inputval, "longtemp")
			else:
				print("Invalid input")

		elif string == "appenddelshortdata":
			if inputval != None:
				dataset.appenddelete(inputval, "shortdata")
			else:
				print("Invalid input")

		elif string == "appenddellongdata":
			if inputval != None:
				dataset.appenddelete(inputval, "longdata")
			else:
				print("Invalid input")
		elif string == "printappenddelshorttemp":
			dataset.print_appenddelete("shorttemp")

		elif string == "printappenddelshortdata":
			dataset.print_appenddelete("shortdata")

		elif string == "printappenddellongtemp":
			dataset.print_appenddelete("longtemp")

		elif string == "printappenddellongdata":
			dataset.print_appenddelete("longdata")

		elif string == "removeappenddelshorttemp":
			if inputval != None:
				dataset.remove_appenddelete(inputval, "shorttemp")
			else:
				print("Invalid input")

		elif string == "removeappenddelshortdata":
			if inputval != None:
				dataset.remove_appenddelete(inputval, "shortdata")
			else:
				print("Invalid input")	

		elif string == "removeappenddellongtemp":
			if inputval != None:
				dataset.remove_appenddelete(inputval, "longtemp")
			else:
				print("Invalid input")	

		elif string == "removeappenddellongdata":
			if inputval != None:
				dataset.remove_appenddelete(inputval, "longdata")
			else:
				print("Invalid input")				
		#elif string == "viewdataelement":
			#if inputval != None:
				#dataset.viewdataelement(inputval)
			#else:
				#print("Invalid input")

		#elif string == "viewtempelement":
			#if inputval != None:
				#dataset.viewtempelement(inputval)
			#else:
				#print("Invalid input")
		elif string == "loadpredictor":
			predictionParameters = predict.setPredictor()

		elif string == "setdatasetfolder":
			if inputval != None:
				dataset.setDatasetFolder(inputval)
			else:
				print("Invalid input")

		elif string == "printdatasetfolder":
			dataset.printDatasetFolder()

		elif string == "exportfft":
			exportThread = threading.Thread(target=dataset.exportPlots, 
												args=("temp", "fft","fast")) 
			exportThread.start()
			#dataset.exportPlots("temp", "fft")
		elif string == "exportraw":
			exportThread = threading.Thread(target=dataset.exportPlots, 
												args=("data", "raw")) 
			exportThread.start()
			#dataset.exportPlots("temp", "raw")
		elif string == "fftplot":
			if inputval != None:
				plotlib.fftplot(inputval)
				plt.show()
			else:	
				plotlib.fftplot(0)
				plt.show()
		
		elif string == "loaddataset":
			x,y = dataset.loadDataset("data.txt")
			#x,y = dataset.sortDataset(x, y, classes=[0,5,2,4,6,8])
			print(x[0][0])
		elif string == "stats":
			dataset.datasetStats("data.txt")	

		elif string == "testsave":
			dataset.saveLongTemp(0)
		
		elif string == "learn":

			learnThread = threading.Thread(target=learning.startLearning, args=())
			learnThread.setDaemon(True)
			learnThread.start()
		
		elif string == "predict":
			print("Starting to load predictor")
			
			#predictionParameters = predict.loadPredictor(machinestate)
			
			predictioncondition = True
			
			#print featuremask
			print("Predictionsetup complete")

		elif string == "notpredict":
			predictioncondition = False

		elif string == "report":

			predict.classificationReportGUI()

		elif string == "clearreport":
			predict.resetClassificationReport()

		elif string == "hht":
			if inputval != None and inputval < 10:
				hht.multiplottestHHT(inputval)
		
		elif string == "drone":
			predictioncondition = True
			predictionThread = threading.Thread(target=predict.predictRealTime,kwargs=(predictionParameters))
			predictionThread.setDaemon(True)
			predictionThread.start()		
			#housekeepThread = threading.Thread(target=controller.housekeeper, args=())
			#housekeepThread.start()
			#controller.droneController()
			controller.originalDroneController()
		elif string == "speak":
			threadSpeak = threading.Thread(target=speak.speakSystem,args=())
			threadSpeak.setDaemon(True)
			threadSpeak.start()
		elif string == "online":
			predictioncondition = True
			predictionThread = threading.Thread(target=predict.predictRealTime,kwargs=(predictionParameters))
			predictionThread.setDaemon(True)
			predictionThread.start()		
			#housekeepThread = threading.Thread(target=controller.housekeeper, args=())
			#housekeepThread.start()
			online = threading.Thread(target=controller.onlineVerificationController, args=())
			online.start()

		elif string == "dronesimulator":		
			housekeepThread = threading.Thread(target=controller.housekeeper, args=())
			housekeepThread.start()
			droneSimulatorThread = threading.Thread(target=controller.droneSimulatorController, args=())
			droneSimulatorThread.start()
		
		elif string == "mujaffa":
			webbrowser.open('http://www.123spill.no/spill/spill-spillet/448/Mujaffa-1.6', new = 2)
			housekeepThread = threading.Thread(target=controller.housekeeper, args=())
			housekeepThread.start()
			mujaffaThread = threading.Thread(target=controller.mujaffaController, args=())
			mujaffaThread.start()

		elif string == "supermario":	
			webbrowser.open('http://nrksuper.no/super/spill/super-mario-flash/', new = 2)
			housekeepThread = threading.Thread(target=controller.housekeeper, args=())
			housekeepThread.start()
			superMarioThread = threading.Thread(target=controller.superMarioController, args=())
			superMarioThread.start()

		elif string == "tetris":
			webbrowser.open('https://tetris.com/play-tetris/', new = 2)
			housekeepThread = threading.Thread(target=controller.housekeeper, args=())
			housekeepThread.start()
			superMarioThread = threading.Thread(target=controller.tetrisController, args=())
			superMarioThread.start()
		
		elif string == "help":
			print("This is a list over essential commands, those ending with = need input values")
			print("exit - exits the system")
			print("start - starts the serial communication with the EEG helmet")
			print("graph - realtime graph with filtered data")
			print("gui - start the training gui")
			print("learn - train the system")
			print("analyzefilter - makes a plot of amplitude, phase, impule and step response for the current filter")
			print("exportdataplots - exports time domain plots for dataset, files can be found in Dataset_exports folder")
			print("exporttempplots - export timedomain plots for temp dataset, files can be found in Dataset_exports folder")
			print("exportraw - export unfiltered timedomain plots for both temp set and data set")
			print("exportfft - export fft plots for both temp set and data set")
			print("helpdataset - print out the dataset handling commands")

			print("+ many more, look in the code in the keys() function in main.py file for more...")
		elif string == "helpdataset":
			print("This is a list over essential dataset commands")
		else:
			print("Unknown command")	

		tme.sleep(0.5)


def save():
	np.savetxt('data.out', data[1][1])

def gui():
	ttk.guiloop()



def main():
	global graphVar, exit, guiVar, predictioncondition

	
	#setpriority(5)
			

	threadHK = threading.Thread(target=housekeeper,args=())
	threadHK.setDaemon(True)
	threadHK.start()
	threadKeys = threading.Thread(target=keys,args=())
	#threadKeys.setDaemon(True)
	threadKeys.start()
	


	print("Setup finished, starting threads")

	if len(sys.argv) > 1:
		if sys.argv[1] == "gui":
			app = ttk.App()
		elif sys.argv[1] == "drone":
			killkeys=True
			threadDataCatcher = threading.Thread(target=dataCatcher,args=())
			threadDataCatcher.setDaemon(True)
			threadDataCatcher.start()
			predictioncondition=True
			housekeepThread = threading.Thread(target=controller.housekeeper, args=())
			housekeepThread.start()
			#controller.droneController()
			controller.originalDroneController()
			os._exit(0)



	#threadDataCatcher = threading.Thread(target=dataCatcher,args=())
	#threadDataCatcher.setDaemon(True)
	#threadDataCatcher.start()
	#threadGui = threading.Thread(target=ttk.guiloop, args=())
	#threadGui.setDaemon(True)
	#threadGui.start()
	#ttk.guiloop()
	#thread2 = threading.Thread(target=QtGui.QApplication.instance().exec_(),args=())

	#thread2.start()
	
	#thread0.join()
	#thread1.join()
	#thread2.join()

	#while not graphVar:
		#tme.sleep(0.1)
	'''	
	while not guiVar:
		tme.sleep(0.1)
	if not exit:
		#graph()	
		#ttk.guiloop()
		app = ttk.App()
	'''
	while not exit:
		tme.sleep(1)

	

if __name__ == '__main__':
	main()