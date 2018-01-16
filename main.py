import sys; sys.path.append('.') # help python find open_bci_v3.py relative to scripts folder
sys.path.append('../OpenBCI')
import open_bci_v3 as bci
import os
import logging
import time as tme
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time
import threading
#from threading import Lock
from scipy import signal
import scipy.fftpack
import matplotlib.pyplot as plt
#from numpy.random import randint
import Tkinter as tk
import gui as ttk
import plot as plotlib
import filterlib 
import dataset
from globalvar import *

#mutex = Lock()
board = None
root = None
graphVar = False
count = None
curves = []
ptr = 0
p = None
exit = False
filtering = True
averageCondition = False

app = QtGui.QApplication([])



def dataCatcher():
	global board
	#Helmetsetup
	port = 'COM6'
	baud = 115200
	logging.basicConfig(filename="test.log",format='%(asctime)s - %(levelname)s : %(message)s',level=logging.DEBUG)
	logging.info('---------LOG START-------------')
	board = bci.OpenBCIBoard(port=port, scaled_output=True, log=True, filter_data = False)
	print("Board Instantiated")
	board.ser.write('v')
	#tme.sleep(10)

	if not board.streaming:
		board.ser.write(b'b')
		board.streaming = True

	print("Samplerate: %0.2fHz" %board.getSampleRate())

	board.start_streaming(printData)


def printData(sample):	
	global nSamples, nPlots, data, df, init, newSamples, rawdata, threadFilter 
	global newTimeData, timeData, timestamp, xt 
	global mutex, window
	
	xt = tme.time()
	
	for i in range(nPlots):			
		newSamples[i].append(sample.channel_data[i])
		newTimeData[i].append(xt)

	if len(data[0][0]) >= nSamples:
		with(mutex):
			for i in range(nPlots):				
				data[i][rawdata].pop(0)
				data[i][filterdata].pop(0)
				data[i][timestamp].pop(0)


	if len(newSamples[0]) >= window:
		#print("Starting filter\n")
		#print(newSamples[0])
						
		#threadFilter = threading.Thread(target=filterlib.filter,args=(newSamples, newTimeData))
		#threadFilter.setDaemon(True)
		#threadFilter.start()
		#threadFilter.join()
		#print("Joining thread\n")
		filterlib.filter(newSamples, newTimeData)
		#print(len(averagedata[0]))
		newTimeData = [],[],[],[],[],[],[],[]
		newSamples = [],[],[],[],[],[],[],[]



def update():
	global curves, data, ptr, p, lastTime, fps, nPlots, count, board
	count += 1
	string = ""
	with(mutex):
		for i in range(nPlots):
			curves[i].setData(data[i][filterdata])
			if len(data[i][filterdata])>100:
				string += '   Ch: %d ' % i
				string += ' = %0.2f uV ' % data[i][filterdata][-1]

	ptr += nPlots
	#now = time()
	#dt = now - lastTime	
	#lastTime = now

	p.setTitle(string)
    #app.processEvents()  ## force complete redraw for every plot

def graph():
	#Graph setup
	global nPlots, nSamples, count, data, curves, p, QtGui, app
	#app = QtGui.QApplication([])
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
		c.setPos(0,i*100+200)
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
	while True:
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
		elif string == "exit":
			print("Initiating exit sequence")
			exit = True
			if root != None:
				root.destroy()
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
			#threadDataCatcher.setDaemon(True)
			threadDataCatcher.start()
		elif string == "graph":
			graphVar = True

		elif string == "gui":
			threadGui = threading.Thread(target=gui, args=())
			threadGui.setDaemon(True)
			threadGui.start()
		elif string == "exportdataplots":
			dataset.exportPlots("data")
		elif string == "exporttempplots":
			dataset.exportPlots("temp")
		elif string == "exportallplots":
			dataset.exportPlots("all")

		elif string == "saveshortdata":
			dataset.saveShortData()

		elif string == "savelongdata":
			dataset.saveLongData()

		elif string == "deleteshortdata":
			dataset.clearShortData()

		elif string == "deletelongdata":
			dataset.clearLongData()

		elif string == "deleteshorttemp":
			dataset.clearShortTemp()

		elif string == "deletelongtemp":
			dataset.clearLongTemp()


		elif string == "savefilter":
			savefiltercoeff()
			

		elif string == "deleteshortdataelement":
			if inputval != None:
				dataset.deleteShortDataelement(inputval)
			else:
				print("Invalid input")

		elif string == "deleteshorttempelement":
			if inputval != None:
				dataset.deleteShortTempelement(inputval)
			else:
				print("Invalid input")

		elif string == "deletelongdataelement":
			if inputval != None:
				dataset.deleteLongDataelement(inputval)
			else:
				print("Invalid input")

		elif string == "deletelongtempelement":
			if inputval != None:
				dataset.deleteLongTempelement(inputval)
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
		elif string == "exportfft":
			dataset.exportPlots("all", "fft")
		elif string == "fftplot":
			plotlib.fftplot(0)
			plt.show()
		elif string == "loaddataset":
			x,y = dataset.loadDataset("temp.txt")
			for i in range(len(x[0])):
				print(len(x[0][0]))
		else:
			print("Unknown command")	
def save():
	np.savetxt('data.out', data[1][1])

def gui():
	ttk.guiloop()



def main():
	global graphVar, exit

	print("Setup finished, starting threads")

	threadKeys = threading.Thread(target=keys,args=())
	threadKeys.setDaemon(True)
	threadKeys.start()
	#threadDataCatcher = threading.Thread(target=dataCatcher,args=())
	#threadDataCatcher.setDaemon(True)
	#threadDataCatcher.start()
	#threadGui = threading.Thread(target=gui, args=())
	#threadGui.setDaemon(True)
	#threadGui.start()
	
	#thread2 = threading.Thread(target=QtGui.QApplication.instance().exec_(),args=())

	#thread2.start()
	
	#thread0.join()
	#thread1.join()
	#thread2.join()

	while not graphVar:
		pass

	if not exit:
		graph()	

	while not exit:
		pass


if __name__ == '__main__':
	main()