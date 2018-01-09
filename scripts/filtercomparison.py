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
from threading import Lock
from scipy import signal
import matplotlib.pyplot as plt
from numpy.random import randn
#import tkinter

mutex = Lock()

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

#Graph setup
app = QtGui.QApplication([])
p = pg.plot()
nPlots = 6
nSamples = 2000
p.setWindowTitle('pyqtgraph example: MultiPlotSpeedTest')
#p.setRange(QtCore.QRectF(0, -10, 5000, 20)) 
p.setLabel('bottom', 'Index', units='B')
#curves = [p.plot(pen=(i,nPlots*1.3)) for i in range(nPlots)]

curves = []
ptr = 0
lastTime = time()
fps = None
count = 0
data = [],[],[],[],[],[],[],[]
rawdata = [],[],[],[],[],[],[],[]
averagedata = [],[],[],[],[],[],[],[]
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



print("Graphsetup finished")


#Filtersetup
#Notchfilter
window = 151

fs = 250.0
f0 = 50.0
Q = 50
w0 = f0/(fs/2)
notchB, notchA = signal.iirnotch(w0, Q) 


filtering = True
bandstopFilter = False
lowpassFilter = False
bandpassFilter = True


sample = board._read_serial_binary()
notchZi = np.zeros([8,2])
init = True

#Butterworth lowpass filter
N  = 4    # Filter order
fk = 30
Wn = fk/(fs/2) # Cutoff frequency
lowpassB, lowpassA = signal.butter(N, Wn, output='ba')
lowpassZi = np.zeros([8,N])


#FIR bandpass filter
hc = 45.0/(fs/2) #High cut
lc = 5.0/(fs/2)	#Low cut

bandpassB = signal.firwin(window, [lc, hc], pass_zero=False, window = 'hann') #Bandpass
bandpassA = 1.0 #np.ones(len(bandpassA))
bandpassZi = np.zeros([8, window-1])

highpassB = signal.firwin(window, lc, pass_zero=False, window = 'hann') #Bandpass
highpassA = 1.0
highpassZi = np.zeros([8, window-1])
print("Filtersetup finished")

multibandB = signal.firwin(window, [lc, hc, hcc], pass_zero=False, window = 'hann') #Bandpass
multibandA = 1.0
multibandZi = np.zeros([8, window-1])

def dataCatcher():
	global board
	board.start_streaming(printData)


def printData(sample):	
	global nPlots, data, df, init, averagedata, rawdata, threadFilter 

	with(mutex):

		for i in range(nPlots):
			avg = 0
			rawdata[i].append(sample.channel_data[i])

			for j in range(len(rawdata[0])-1):
				avg = avg + rawdata[i][j]

			average = avg / (len(rawdata[0]))

			if filtering:
				averagedata[i].append(sample.channel_data[i]-average)
				#averagedata[i].append(sample.channel_data[i])
			else:
				data[i].append(sample.channel_data[i])
				
		if len(data[0]) >= nSamples:
			for i in range(nPlots):
				data[i].pop(0)
				
				
		if len(rawdata[0]) > 1000:
			for i in range(nPlots):
				rawdata[i].pop(0)

		if len(averagedata[0]) >= window:					
			threadFilter = threading.Thread(target=notchFilter,args=())
			threadFilter.start()

def appendData(y, i):
	for j in range(len(y)):
		data[i].append(y[j]*df)

def notchFilter():
	global highpassB, highpassA, highpassZi
	global lowpassB, lowpassA, lowpassZi 
	global bandpassB, bandpassA, bandpassZi
	global notchB, notchA, notchZi 
	global averagedata, data, window, init, initNotch, initLowpass, initBandpass
	global bandstopFilter, lowpassFilter, bandpassFilter
	global multibandB, multibandA, multibandZi
	with(mutex):
		
		if init == True: #Gjor dette til en funksjon, input koeff, return zi
			for i in range(nPlots):
				notchZi[i] = signal.lfilter_zi(notchB, notchA) * averagedata[i][0]
				lowpassZi[i] = signal.lfilter_zi(lowpassB, lowpassA) * averagedata[i][0]
				bandpassZi[i] = signal.lfilter_zi(bandpassB, bandpassA) * averagedata[i][0]
				highpassZi[i] = signal.lfilter_zi(highpassB, highpassA) * averagedata[i][0]
				multibandZi[i] = signal.lfilter_zi(multibandB, multibandA) * averagedata[i][0]
			init = False

			#TODO: init filters again when turned on
		for i in range(2):
			x = averagedata[i]
			x, bandpassZi[i] = signal.lfilter(bandpassB, bandpassA, x, zi=bandpassZi[i])	
			appendData(x,i)
		
		for i in range(2):
			x = averagedata[i+2]
			x, notchZi[i] = signal.lfilter(notchB, notchA, x, zi=notchZi[i])
			x, highpassZi[i] = signal.lfilter(highpassB, highpassA, x, zi=highpassZi[i])
			appendData(x,i+2)

		for i in range (2):
			x = averagedata[i+4]
			x, notchZi[i+2] = signal.lfilter(notchB, notchA, x, zi=notchZi[i+2])
			appendData(x,i+4)
		averagedata = [],[],[],[],[],[],[],[]

def plot():
	with(mutex):
		while len(data[0]) > nSamples:
				for i in range(nPlots):
					data[i].pop(0)
		legends = []
		plt.figure(1)
		x = np.arange(0, len(data[1])/fs, 1/fs)
		
		for i in range(2):
			label = "Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			ax1 = plt.subplot(311)
			legend, = plt.plot(x, data[i], label=label)
			legends.append(legend)
		ax1.set_title("Bandpass")
		plt.legend(handles=legends)
		#ax1.ylabel('uV')
		for i in range(2):
			label = "Notch + HP Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			ax2 = plt.subplot(312)
			legend, = plt.plot(x, data[i+2], label=label)
			#legends.append(legend)
		ax2.set_title("Notch + Highpass")
		#ax2.ylabel('uV')
		for i in range(2):
			label = "Notch Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			ax3 = plt.subplot(313)
			legend, = plt.plot(x, data[i+4], label=label)
			#legends.append(legend)
		ax3.set_title("Notch")
		#ax3.ylabel('uV')
	plt.ylabel('uV')
	plt.xlabel('Seconds')

	#TODO: x axis in seconds
	#plt.legend(handles=legends)
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
	global curve, data, ptr, p, lastTime, fps, nPlots, count, board
	count += 1
	string = ""
	with(mutex):
		for i in range(nPlots):
			curves[i].setData(data[i])
			if len(data[i])>100:
				string += '   Ch: %d ' % i
				string += ' = %0.2f uV ' % data[i][-1]

	ptr += nPlots
	now = time()
	dt = now - lastTime	
	lastTime = now

	p.setTitle(string)
    #app.processEvents()  ## force complete redraw for every plot

def keys():

	global board, bandstopFilter, filtering, lowpassFilter, bandpassFilter
	while True:
		string = raw_input()
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
			board.stop()
			board.disconnect()
			QtGui.QApplication.quit()
			sys.exit()
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

#def gui():
	#root = Tk()

	#w = Label(root, text="Hello Tkinter!")
	#w.pack()

	#root.mainloop()	


def main():

	mw = QtGui.QMainWindow()
	mw.resize(800,800)
	
	timer = QtCore.QTimer()
	timer.timeout.connect(update)
	timer.start(0)

	print("Setup finished, starting threads")

	threadKeys = threading.Thread(target=keys,args=())
	threadDataCatcher = threading.Thread(target=dataCatcher,args=())
	#threadGui = threading.Thread(target=gui, args=())
	threadKeys.start()
	threadDataCatcher.start()
	#threadGui.start()	
	
	#thread2 = threading.Thread(target=QtGui.QApplication.instance().exec_(),args=())

	#thread2.start()
	
	#thread0.join()
	#thread1.join()
	#thread2.join()
	
	QtGui.QApplication.instance().exec_()

	


if __name__ == '__main__':
	main()
