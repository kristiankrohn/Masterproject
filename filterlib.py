
from scipy import signal
import scipy.fftpack
import numpy as np
from globalvar import *


window = 3
#fs = 250.0
f0 = 50.0
Q = 50
w0 = f0/(fs/2)
notchB, notchA = signal.iirnotch(w0, Q) 
f100 = 100.0
w100 = f100/(fs/2)
bNotch100, aNotch100 = signal.iirnotch(w100, Q)

DcNotchA = [1 , -0.9] 
DcNotchB = [1,-1]

notchZi = np.zeros([8,2])
notchZi2 = np.zeros([8,2])
notchZi3 = np.zeros([8,2])
notchZi4 = np.zeros([8,2])
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

def appendData(y, i, xt, yr):
	global data, rawdata
	global avgTimeData, timeData
	for j in range(len(y)):
		data[i].append(y[j])
		timeData[i].append(xt[j])
		rawdata[i].append(yr[j])

def init_filter():
	global lowpassB, lowpassA, lowpassZi 
	global bandpassB, bandpassA, bandpassZi
	global notchB, notchA, notchZi, notchZi2, notchZi3, notchZi4 
	global newSamples
	global bandstopFilter, lowpassFilter, bandpassFilter
	global DcNotchB, DcNotchA, DCnotchZi, DCnotchZi2
	global bNotch100, aNotch100, notchZi1001, notchZi1002

	if init == True: #Dette er bare tull
		
		for i in range(nPlots):
			notchZi[i] = signal.lfilter_zi(notchB, notchA) * newSamples[i][0]
			notchZi2[i] = signal.lfilter_zi(notchB, notchA) * newSamples[i][0]
			notchZi3[i] = signal.lfilter_zi(notchB, notchA) * newSamples[i][0]
			notchZi4[i] = signal.lfilter_zi(notchB, notchA) * newSamples[i][0]
			notchZi1001[i] = signal.lfilter_zi(bNotch100, aNotch100) * newSamples[i][0]
			notchZi1002[i] = signal.lfilter_zi(bNotch100, aNotch100) * newSamples[i][0]
			lowpassZi[i] = signal.lfilter_zi(lowpassB, lowpassA) * newSamples[i][0]
			bandpassZi[i] = signal.lfilter_zi(bandpassB, bandpassA) * newSamples[i][0]
			DCnotchZi[i] = signal.lfilter_zi(DcNotchB, DcNotchA) * newSamples[i][0]
			DCnotchZi2[i] = signal.lfilter_zi(DcNotchB, DcNotchA) * newSamples[i][0]
		init = False


def filter():
	global lowpassB, lowpassA, lowpassZi 
	global bandpassB, bandpassA, bandpassZi
	global notchB, notchA, notchZi, notchZi2, notchZi3 
	global newSamples, data, window, init, initNotch, initLowpass, initBandpass
	global bandstopFilter, lowpassFilter, bandpassFilter
	global avgTimeData, timeData, mutex
	global DcNotchB, DcNotchA, DCnotchZi, DCnotchZi2
	global bNotch100, aNotch100, notchZi1001, notchZi1002


	with(mutex):
			
		for i in range(nPlots):
			x = newSamples[i]
			yr = x
			xt = avgTimeData[i]

			x, DCnotchZi[i] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi[i])
			#x, DCnotchZi2[i] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi2[i])

			if bandstopFilter: #Dette er feil filter
				x, notchZi[i] = signal.lfilter(notchB, notchA, x, zi=notchZi[i])
				x, notchZi2[i] = signal.lfilter(notchB, notchA, x, zi=notchZi2[i])
				x, notchZi3[i] = signal.lfilter(notchB, notchA, x, zi=notchZi3[i])
				x, notchZi4[i] = signal.lfilter(notchB, notchA, x, zi=notchZi4[i])
				x, notchZi1001[i] = signal.lfilter(bNotch100, aNotch100, x, zi=notchZi1001[i])
				x, notchZi1002[i] = signal.lfilter(bNotch100, aNotch100, x, zi=notchZi1002[i])
			if lowpassFilter:
				x, lowpassZi[i] = signal.lfilter(lowpassB, lowpassA, x, zi=lowpassZi[i])
			
			if bandpassFilter:
				x, bandpassZi[i] = signal.lfilter(bandpassB, bandpassA, x, zi=bandpassZi[i])


			appendData(x,i,xt,yr)

		avgTimeData = [],[],[],[],[],[],[],[]
		newSamples = [],[],[],[],[],[],[],[]
