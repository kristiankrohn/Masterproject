import sys; sys.path.append('..') # help python find open_bci_v3.py relative to scripts folder
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
from numpy.random import randint
import Tkinter as tk
import testtkinter as ttk
from globalvar import *


#mutex = Lock()
board = None
root = None
graphVar = False
nPlots = 8

count = None
#Filtersetup
#Notchfilter
#window = 151
window = 3
curves = []
ptr = 0
#data = [],[],[],[],[],[],[],[]

rawdata = [],[],[],[],[],[],[],[]
averagedata = [],[],[],[],[],[],[],[]
avgTimeData = [],[],[],[],[],[],[],[]
averageShortData = [],[],[],[],[],[],[],[]
average = np.zeros(nPlots)
averageShort = np.zeros(nPlots)
p = None

init = True
exit = False
filtering = True
averageCondition = False
bandstopFilter = True
lowpassFilter = False
bandpassFilter = False
app = QtGui.QApplication([])
fs = 250.0
f0 = 50.0
Q = 50
w0 = f0/(fs/2)
notchB, notchA = signal.iirnotch(w0, Q) 
f100 = 100.0
w100 = f100/(fs/2)
bNotch100, aNotch100 = signal.iirnotch(w100, Q)

#sample = board._read_serial_binary()
notchZi = np.zeros([8,2])
notchZi2 = np.zeros([8,2])
notchZi3 = np.zeros([8,2])
notchZi1001 = np.zeros([8,2])
notchZi1002 = np.zeros([8,2])
#print(notchB)
#print(notchA)
#Butterworth lowpass filter
N  = 4    # Filter order
fk = 30
Wn = fk/(fs/2) # Cutoff frequency
lowpassB, lowpassA = signal.butter(N, Wn, output='ba')
lowpassZi = np.zeros([8,N])
DCnotchZi = np.zeros([8,1])
DCnotchZi2 = np.zeros([8,1])

#FIR bandpass filter
hcc = 56.0/(fs/2) #highest cut, only used if multibandpass
hc = 45.0/(fs/2) #High cut
lc = 5.0/(fs/2)	#Low cut

bandpassB = signal.firwin(window, hc, pass_zero=True, window = 'hann') #Bandpass
#bandpassB = signal.firwin(window, [lc, hc], pass_zero=False, window = 'hann') #Bandpass
bandpassA = 1.0 #np.ones(len(bandpassA))
bandpassZi = np.zeros([8, window-1])
#print(bandpassB)
highpassB = signal.firwin(window, lc, pass_zero=False, window = 'hann') #Bandpass
print("Filtersetup finished")

#print("Placeholder %d testing" % hcc)

#GUI parameters
#size = 1000
#speed = 20
#ballsize = 30



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
	global nPlots, data, df, init, averagedata, rawdata, threadFilter 
	global avgTimeData, timeData, timestamp 
	global average, avgLength, mutex
	global averageShortData, avgShortLength
	global averageCondition
	with(mutex):
		timestamp = tme.time()
		#print(timestamp)
		for i in range(nPlots):
			#avg = 0
			if averageCondition:
				rawdata[i].append(sample.channel_data[i])
				if len(rawdata[i]) == avgLength:
					average[i] = average[i] + (rawdata[i][-1]/avgLength) - (rawdata[i][0]/avgLength)
					#print(average[i])
					rawdata[i].pop(0)
				else:
					average[i] = sum(rawdata[i])/len(rawdata[i])
					while len(rawdata[i]) >= avgLength:
						rawdata[i].pop(0)
						print("Faen, abort abort abort!")




				averageShortData[i].append(sample.channel_data[i]-average[i])
				
				if len(averageShortData[i]) == avgShortLength:
					averageShort[i] = averageShort[i] + (averageShortData[i][-1]/avgShortLength) - (averageShortData[i][0]/avgShortLength)
					#print(average[i])
					averageShortData[i].pop(0)
				else:
					averageShort[i] = sum(averageShortData[i])/len(averageShortData[i])
					while len(averageShortData[i]) >= avgShortLength:
						averageShortData[i].pop(0)
						print("Faen, abort abort abort!")					
			else:
				average = np.zeros(nPlots)
				averageShort = np.zeros(nPlots)
			#print(averageShort[i])


			if filtering:
				#averagedata[i].append((sample.channel_data[i]-average[i])-averageShort[i])
				#averagedata[i].append(sample.channel_data[i]-average[i])
				averagedata[i].append(sample.channel_data[i])
				avgTimeData[i].append(timestamp)
			
			else:
				data[i].append(sample.channel_data[i])
				timeData[i].append(timestamp)

		if len(data[0]) >= nSamples:
			for i in range(nPlots):
				data[i].pop(0)
				timeData[i].pop(0)
				
		#if len(rawdata[0]) > 1000:
			#for i in range(nPlots):
				#rawdata[i].pop(0)

	if len(averagedata[0]) >= window:					
		threadFilter = threading.Thread(target=filter,args=())
		#threadFilter.setDaemon(True)
		threadFilter.start()
		#print(len(averagedata[0]))

def appendData(y, i, xt):
	global data
	global avgTimeData, timeData
	for j in range(len(y)):
		data[i].append(y[j])
		timeData[i].append(xt[j])

def filter():
	global lowpassB, lowpassA, lowpassZi 
	global bandpassB, bandpassA, bandpassZi
	global notchB, notchA, notchZi, notchZi2, notchZi3 
	global averagedata, data, window, init, initNotch, initLowpass, initBandpass
	global bandstopFilter, lowpassFilter, bandpassFilter
	global avgTimeData, timeData, mutex
	global DCnotchZi, DCnotchZi2
	global bNotch100, aNotch100, notchZi1001, notchZi1002
	DcNotchA = [1 , -0.9] 
	DcNotchB = [1,-1]
	with(mutex):
		
		if init == True: #Gjor dette til en funksjon, input koeff, return zi
			
			for i in range(nPlots):
				notchZi[i] = signal.lfilter_zi(notchB, notchA) * averagedata[i][0]
				notchZi2[i] = signal.lfilter_zi(notchB, notchA) * averagedata[i][0]
				notchZi3[i] = signal.lfilter_zi(notchB, notchA) * averagedata[i][0]
				notchZi1001[i] = signal.lfilter_zi(bNotch100, aNotch100) * averagedata[i][0]
				notchZi1002[i] = signal.lfilter_zi(bNotch100, aNotch100) * averagedata[i][0]
				lowpassZi[i] = signal.lfilter_zi(lowpassB, lowpassA) * averagedata[i][0]
				bandpassZi[i] = signal.lfilter_zi(bandpassB, bandpassA) * averagedata[i][0]
				DCnotchZi[i] = signal.lfilter_zi(DcNotchB, DcNotchA) * averagedata[i][0]
				DCnotchZi2[i] = signal.lfilter_zi(DcNotchB, DcNotchA) * averagedata[i][0]
			init = False

			#TODO: init filters again when turned on
		for i in range(nPlots):
			x = averagedata[i]
			xt = avgTimeData[i]

			x, DCnotchZi[i] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi[i]);
			#x, DCnotchZi2[i] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi2[i]);

			if bandstopFilter:
				x, notchZi[i] = signal.lfilter(notchB, notchA, x, zi=notchZi[i])
				x, notchZi2[i] = signal.lfilter(notchB, notchA, x, zi=notchZi2[i])
				x, notchZi3[i] = signal.lfilter(notchB, notchA, x, zi=notchZi3[i])
				x, notchZi1001[i] = signal.lfilter(notchB, notchA, x, zi=notchZi1001[i])
				x, notchZi1002[i] = signal.lfilter(notchB, notchA, x, zi=notchZi1002[i])
			if lowpassFilter:
				x, lowpassZi[i] = signal.lfilter(lowpassB, lowpassA, x, zi=lowpassZi[i])
			
			if bandpassFilter:
				
				x, DCnotchZi[i] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi[i]);
				x, DCnotchZi2[i] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi2[i]);
				x, bandpassZi[i] = signal.lfilter(bandpassB, bandpassA, x, zi=bandpassZi[i])


			appendData(x,i,xt)


		averagedata = [],[],[],[],[],[],[],[]
		avgTimeData = [],[],[],[],[],[],[],[]

def plot():
	global mutex
	with(mutex):
		while len(data[0]) > nSamples:
			for i in range(nPlots):
				data[i].pop(0)
		x = np.arange(0, len(data[1])/fs, 1/fs)
		legends = []
		for i in range(2):
			label = "Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			legend, = plt.plot(x, data[i], label=label)
			legends.append(legend)
	plt.ylabel('uV')
	plt.xlabel('Seconds')
	plt.legend(handles=legends)
	legends = []
	plt.show()

def plotAll():
	legends = []
	for i in range(nPlots):
		label = "Channel %d" %(i+1)
		#print(label)
		#label = tuple([label])
		legend, = plt.plot(data[i], label=label)
		legends.append(legend)
	plt.ylabel('uV')
	plt.xlabel('Sample')
	plt.legend(handles=legends)
	legends = []
	plt.show()

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

def fftplot(channel, title=""):
	global fs
	y = data[channel]
	# Number of samplepoints
	N = len(y)
	# sample spacing
	T = 1.0 / fs
	x = np.linspace(0.0, N*T, N)
	yf = scipy.fftpack.fft(y)
	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

	fig, ax = plt.subplots()
	ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
	ax.set_title(title)
	plt.ylabel('uV')
	plt.xlabel('Frequency (Hz)')

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