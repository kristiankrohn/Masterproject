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
import scipy.fftpack
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
nPlots = 8
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
averagedata = [],[],[],[],[],[],[],[],[],[],[]
displayUV = []
df = 1

averageShortData = [],[],[],[],[],[],[],[]
averageShortDataAlone = [],[],[],[],[],[],[],[]
average = np.zeros(nPlots)
averageShort = np.zeros(nPlots)
averageShortAlone = np.zeros(nPlots)
avgLength = 50
avgShortLength = 25


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
f100 = 100.0
w100 = f100/(fs/2)
bNotch100, aNotch100 = signal.iirnotch(w100, Q)
notchZi100 = np.zeros([8,2])

sample = board._read_serial_binary()
notchZi = np.zeros([8,2])
notchZi2 = np.zeros([8,2])
init = True
DCnotchZi = np.zeros([8,1])
DCnotchZi2 = np.zeros([8,1])

#Butterworth lowpass filter
N  = 4    # Filter order
fk = 30
Wn = fk/(fs/2) # Cutoff frequency
lowpassB, lowpassA = signal.butter(N, Wn, output='ba')
lowpassZi = np.zeros([8,N])


#FIR bandpass filter
hcc = 56.0/(fs/2)
hc = 44.0/(fs/2) #High cut
lc = 5.0/(fs/2)	#Low cut

bandpassB = signal.firwin(window, [lc, hc], pass_zero=False, window = 'hann') #Bandpass
bandpassA = 1.0 #np.ones(len(bandpassA))
bandpassZi = np.zeros([8, window-1])

highpassB = signal.firwin(window, lc, pass_zero=False, window = 'hann') #Bandpass
highpassA = 1.0
highpassZi = np.zeros([8, window-1])
print("Filtersetup finished")

multibandB = signal.firwin(window, hc, pass_zero=True, window = 'hann') #Bandpass
multibandA = 1.0
multibandZi = np.zeros([8, window-1])

def dataCatcher():
	global board
	board.start_streaming(printData)


def printData(sample):	
	global nPlots, data, df, init, averagedata, rawdata, threadFilter 
	global averageShortData, avgShortLength, average, avgLength
	with(mutex):

		for i in range(nPlots):
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

			averageShortDataAlone[i].append(sample.channel_data[i])

			if len(averageShortDataAlone[i]) == avgShortLength:
				averageShortAlone[i] = averageShortAlone[i] + (averageShortDataAlone[i][-1]/avgShortLength) - (averageShortDataAlone[i][0]/avgShortLength)
				#print(average[i])
				averageShortDataAlone[i].pop(0)
			else:
				averageShortAlone[i] = sum(averageShortDataAlone[i])/len(averageShortDataAlone[i])
				while len(averageShortDataAlone[i]) >= avgShortLength:
					averageShortDataAlone[i].pop(0)
					print("Faen, abort abort abort!")
			#if i == 0:
				#data[i].append(sample.channel_data[i])
			if filtering:
				averagedata[i].append(sample.channel_data[i]-average[i])
				if i == 0:
					averagedata[8].append(sample.channel_data[i])
					averagedata[9].append(sample.channel_data[i]-averageShort[i]-average[i])
					averagedata[10].append(sample.channel_data[i]-averageShortAlone[i])
			else:
				data[i].append(sample.channel_data[i])
				
		if len(data[1]) >= nSamples:
			#print(data)
			for i in range(nPlots):
				try:
					data[i].pop(0)
				except IndexError:
					print("Faen")
				
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
	global DCnotchZi, DCnotchZi2, notchZi2
	global bNotch100, aNotch100, notchZi100
	DcNotchA = [1 , -0.98] 
	DcNotchB = [1,-1]
	with(mutex):
		
		if init == True: #Gjor dette til en funksjon, input koeff, return zi
			for i in range(nPlots):
				notchZi100[i] = signal.lfilter_zi(bNotch100, aNotch100) * averagedata[i][0]
				notchZi[i] = signal.lfilter_zi(notchB, notchA) * averagedata[i][0]
				notchZi2[i] = signal.lfilter_zi(notchB, notchA) * averagedata[i][0]
				lowpassZi[i] = signal.lfilter_zi(lowpassB, lowpassA) * averagedata[i][0]
				bandpassZi[i] = signal.lfilter_zi(bandpassB, bandpassA) * averagedata[i][0]
				highpassZi[i] = signal.lfilter_zi(highpassB, highpassA) * averagedata[i][0]
				multibandZi[i] = signal.lfilter_zi(multibandB, multibandA) * averagedata[i][0]
				DCnotchZi[i] = signal.lfilter_zi(DcNotchB, DcNotchA) * averagedata[i][0]
				DCnotchZi2[i] = signal.lfilter_zi(DcNotchB, DcNotchA) * averagedata[i][0]
			init = False
		#appendData(averagedata[10],3) #Short average
		#appendData(averagedata[9],2) #Dual average
		appendData(averagedata[8],0) #Raw Data
		appendData(averagedata[0],1) #Long Average
			#TODO: init filters again when turned on
		for i in range(1):
			x = averagedata[0] #DC notch
			#x, notchZi[i+6] = signal.lfilter(notchB, notchA, x, zi=notchZi[i+6])
			x, DCnotchZi[i] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi[i]);
			x, notchZi[i+2] = signal.lfilter(notchB, notchA, x, zi=notchZi[i+2])
			#x, bandpassZi[i] = signal.lfilter(bandpassB, bandpassA, x, zi=bandpassZi[i])	
			appendData(x,i+2)
		
		for i in range(1):
			x = averagedata[i] #Dual DC notch
			#x, notchZi[i] = signal.lfilter(notchB, notchA, x, zi=notchZi[i])
			x, DCnotchZi[i+3] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi[i+3]);
			x, DCnotchZi2[i+3] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi2[i+3]);
			x, notchZi[i+3] = signal.lfilter(notchB, notchA, x, zi=notchZi[i+3])
			#x, highpassZi[i] = signal.lfilter(highpassB, highpassA, x, zi=highpassZi[i])
			appendData(x,i+3)

		for i in range(1):
			x = averagedata[9] #Dual average notch
			x, notchZi[i+4] = signal.lfilter(notchB, notchA, x, zi=notchZi[i+4])
			x, notchZi2[i+4] = signal.lfilter(notchB, notchA, x, zi=notchZi2[i+4])
			#x, multibandZi[i] = signal.lfilter(multibandB, multibandA, x, zi=multibandZi[i])
			#x, highpassZi[i+2] = signal.lfilter(highpassB, highpassA, x, zi=highpassZi[i+2])
			appendData(x,i+4)

		for i in range (1):
			x = averagedata[1] # Highpass Notch Average
			#x = averagedata[0] # Highpass Notch
			x, highpassZi[i+5] = signal.lfilter(highpassB, highpassA, x, zi=highpassZi[i+5])
			x, notchZi[i+5] = signal.lfilter(notchB, notchA, x, zi=notchZi[i+5])
			appendData(x,i+5)

		for i in range(1):
			x = averagedata[1] #Bandpass average
			#x, multibandZi[i+2] = signal.lfilter(multibandB, multibandA, x, zi=multibandZi[i+2])
			x, bandpassZi[i+6] = signal.lfilter(bandpassB, bandpassA, x, zi=bandpassZi[i+6])	
			appendData(x,i+6)

		for i in range(1):
			x = averagedata[10]
			x, DCnotchZi[i+7] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi[i+7]);
			#x, DCnotchZi2[i+7] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi2[i+7]);
			x, notchZi[i+7] = signal.lfilter(notchB, notchA, x, zi=notchZi[i+7])
			x, notchZi2[i+7] = signal.lfilter(notchB, notchA, x, zi=notchZi2[i+7])
			x, notchZi2[i+6] = signal.lfilter(notchB, notchA, x, zi=notchZi2[i+6])
			x, notchZi100[i+7] = signal.lfilter(bNotch100, aNotch100, x, zi=notchZi100[i+7])
			#x, highpassZi[i+4] = signal.lfilter(highpassB, highpassA, x, zi=highpassZi[i+4])
			
			appendData(x,i+7) #Short average notch
		averagedata = [],[],[],[],[],[],[],[],[],[],[]

def plot():
	with(mutex):
		#while len(data[0]) > nSamples:
			#data[0].pop(0)
		#while len(data[6]) > nSamples:
			#data[6].pop(0)
		#while len(data[7]) > nSamples:
			#data[7].pop(0)
		for i in range(nPlots):
			print(len(data[i]))

		while len(data[0]) > nSamples:
			for i in range(nPlots):
				data[i].pop(0)
		legends = []
		plt.figure(1)
		x = np.arange(0, len(data[1])/fs, 1/fs)
		
		for i in range(1):
			label = "X %d" %(i+1)
			#print(label)
			#label = tuple([label])
			ax1 = plt.subplot(421)
			legend, = plt.plot(x, data[i], label=label)
			legends.append(legend)
		ax1.set_title("Raw data")
		plt.legend(handles=legends)
		plt.ylabel('uV')
		plt.xlabel('Seconds')
		#ax1.ylabel('uV')
		for i in range(1):
			#label = "Notch + HP Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			ax2 = plt.subplot(423)
			legend, = plt.plot(x, data[i+1], label=label)
			#legends.append(legend)
		ax2.set_title("Raw data - Average, N = 50")
		plt.ylabel('uV')
		plt.xlabel('Seconds')
		#ax2.ylabel('uV')
		for i in range(1):
			label = "Notch Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			ax3 = plt.subplot(425)
			legend, = plt.plot(x, data[i+2], label=label)
			#legends.append(legend)
		ax3.set_title("DC Notch + 50Hz Notch")
		plt.ylabel('uV')
		plt.xlabel('Seconds')
		for i in range(1):
			label = "Notch Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			ax4 = plt.subplot(427)
			legend, = plt.plot(x, data[i+3], label=label)
			#legends.append(legend)
		ax4.set_title("Dual DC Notch + 50Hz notch")
		plt.ylabel('uV')
		plt.xlabel('Seconds')
		for i in range(1):
			label = "Notch Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			ax5 = plt.subplot(422)
			legend, = plt.plot(x, data[i+4], label=label)
			#legends.append(legend)
		ax5.set_title("Raw data - Dual Average(N=50,25) + Dual 50Hz Notch")
		plt.ylabel('uV')
		plt.xlabel('Seconds')
		for i in range(1):
			label = "Notch Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			ax6 = plt.subplot(424)
			legend, = plt.plot(x, data[i+5], label=label)
			#legends.append(legend)
		ax6.set_title("Average + Highpass + Notch(50Hz)")

		#ax6.set_title("Highpass + Notch(50Hz)")
		plt.ylabel('uV')
		plt.xlabel('Seconds')
		
		for i in range(1):
			#label = "Notch Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			ax7 = plt.subplot(426)
			legend, = plt.plot(x, data[i+6], label=label)
			legends.append(legend)
		
		ax7.set_title("Average + Bandpass")
		plt.ylabel('uV')
		plt.xlabel('Seconds')
		for i in range(1):
			label = "Notch Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			ax8 = plt.subplot(428)
			legend, = plt.plot(x, data[i+7], label=label)
			#legends.append(legend)
		ax8.set_title("Septa Notch (0Hz + 4x50Hz + 2x100Hz)")
		#ax8.set_title("Short average")
		#ax3.ylabel('uV')
	plt.ylabel('uV')
	plt.xlabel('Seconds')

	#TODO: x axis in seconds
	#plt.legend(handles=legends)
	legends = []
	plt.subplots_adjust(hspace=0.6)
	plt.show()

def fftall():
	with(mutex):
		while len(data[0]) > nSamples:
			for i in range(nPlots):
				data[i].pop(0)
		plt.figure(1)
		ax = plt.subplot(421)
		fftplot(0, "Raw data", ax)
		
		ax = plt.subplot(423)
		fftplot(1, "Raw data - Average, N = 50", ax)
		
		ax = plt.subplot(425)
		fftplot(2, "DC Notch + 50Hz Notch", ax)
		
		#plt.subplot(427)
		#fftplot(3, "Dual DC Notch + 50Hz notch")
		
		ax = plt.subplot(422)
		fftplot(4, "Raw data - Dual Average(N=50,25) + Dual 50Hz Notch", ax)
		
		ax = plt.subplot(424)
		fftplot(5, "Average + Highpass + Notch(50Hz)", ax)
		#fftplot(5, "Highpass + Notch(50Hz)", ax)
		ax = plt.subplot(426)
		fftplot(6, "Average + Bandpass", ax)
		
		ax = plt.subplot(428)
		fftplot(7, "Septa Notch (0Hz + 4x50Hz + 2x100Hz)", ax)		
		
		plt.subplots_adjust(hspace=0.6)
		plt.show()


def fftplot(channel, title="", ax=None):
	global fs
	y = data[channel]
	# Number of samplepoints
	N = len(y)
	# sample spacing
	T = 1.0 / fs
	x = np.linspace(0.0, N*T, N)
	from scipy.signal import blackman
	w = blackman(N)
	yf = scipy.fftpack.fft(y*w)
	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

	if ax == None:
		fig, ax = plt.subplots()

	#print(yf)
	yf = 20 * np.log10(2.0/N * (np.abs(yf[:N//2])))
	#print(yf)
	#ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
	ax.plot(xf, yf)
	ax.set_title(title)
	plt.ylabel('dBuV')
	plt.xlabel('Frequency (Hz)')
	#plt.yscale('log')

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
		elif string == "fftall":
			fftall()
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