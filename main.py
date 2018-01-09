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
from globalvar import *
import plot as plotlib
from filterlib import *

#mutex = Lock()
board = None
root = None
graphVar = False
nPlots = 8

count = None

curves = []
ptr = 0

p = None

init = True
exit = False
filtering = True
averageCondition = False
bandstopFilter = True
lowpassFilter = False
bandpassFilter = False
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


def printData(sample):	#This function is too slow, we are loosing data and fucking up everything
	global nPlots, data, df, init, newSamples, rawdata, threadFilter 
	global avgTimeData, timeData, timestamp 
	global mutex

	with(mutex):
		timestamp = tme.time()
		
		for i in range(nPlots):			
			newSamples[i].append(sample.channel_data[i])
			timeData[i].append(timestamp)

		if len(data[0]) >= nSamples:
			for i in range(nPlots):
				data[i].pop(0)
				timeData[i].pop(0)
				rawdata[i].pop(0)

		if len(averagedata[0]) >= window:					
			threadFilter = threading.Thread(target=filter,args=())
			#threadFilter.setDaemon(True)
			threadFilter.start()
			#print(len(averagedata[0]))




def update():
	global curves, data, ptr, p, lastTime, fps, nPlots, count, board
	count += 1
	string = ""
	with(mutex):
		for i in range(nPlots):
			curves[i].setData(data[i])
			if len(data[i])>100:
				string += '   Ch: %d ' % i
				string += ' = %0.2f uV ' % data[i][-1]

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
			plot()
			#plotThread = threading.Thread(target=plot,args=())
			#plotThread.start()
			#plotthread.join()
		elif string == "plotall":
			plotAll()
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
		elif string == "printdata":
			ttk.openFile()
		
		elif string == "save":
			ttk.saveData()

		elif string == "deletealldata":
			ttk.clearData()
		elif string == "cleartemp":
			ttk.clearTemp()
		elif string == "savefilter":
			np.savetxt('bandpasscoeff.out', bandpassB)
			np.savetxt('highpasscoeff.out', highpassB)
			print("Saved filter coefficients")
		elif string == "deletedataelement":
			if inputval != None:
				ttk.deletedataelement(inputval)
			else:
				print("Invalid input")
		elif string == "deletetempelement":
			if inputval != None:
				ttk.deletetempelement(inputval)
			else:
				print("Invalid input")
		elif string == "viewdataelement":
			if inputval != None:
				ttk.viewdataelement(inputval)
			else:
				print("Invalid input")
		elif string == "viewtempelement":
			if inputval != None:
				ttk.viewtempelement(inputval)
			else:
				print("Invalid input")
		elif string == "printtemp":
			ttk.opentemp()
		elif string == "fftplot":
			fftplot(0)
			plt.show()

def save():
	np.savetxt('data.out', data[1])

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