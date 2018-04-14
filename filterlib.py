from scipy import signal
import scipy.fftpack
import numpy as np
from globalconst import*
import globalvar as glb
import threading
import matplotlib.pyplot as plt
import plot





def filter(newSamples, newTimeData, i, b, a):
	#global Zi
	x = newSamples[i]
	xr = x
	xt = newTimeData[i]

	try:
		x, glb.Zi[i] = signal.lfilter(b, a, x, zi=glb.Zi[i])
	except Exception:
		print("Error in filter, returning")
		return
	#x = signal.lfilter(b, a, x)
	for j in range(len(x)):

		glb.data[i][rawdata].append(xr[j])
		glb.data[i][filterdata].append(x[j])
		glb.data[i][timestamp].append(xt[j])

def savefiltercoeff():
	np.savetxt('bcoeff.out', glb.b)
	np.savetxt('acoeff.out', glb.a)
	print("Saved filter coefficients")

def designfilter(Q=50, filtertype="notch"):
	a = [1 , -0.9] 
	b = [1,-1]
	#print("Designing filter")
	#print(glb.fs)
	if filtertype == "notch": 
		f50 = 50.0
		w50 = f50/(glb.fs/2)
		bNotch50, aNotch50 = signal.iirnotch(w50, Q)

		bTot = signal.convolve(b, bNotch50, mode='full')
		bTot = signal.convolve(bTot, bNotch50, mode='full')
		bTot = signal.convolve(bTot, bNotch50, mode='full')
		bTot = signal.convolve(bTot, bNotch50, mode='full')

		aTot = signal.convolve(a, aNotch50, mode='full')
		aTot = signal.convolve(aTot, aNotch50, mode='full')
		aTot = signal.convolve(aTot, aNotch50, mode='full')
		aTot = signal.convolve(aTot, aNotch50, mode='full')

		if glb.fs > 200:
			f100 = 100.0
			w100 = f100/(glb.fs/2)
			bNotch100, aNotch100 = signal.iirnotch(w100, Q)

			bTot = signal.convolve(bTot, bNotch100, mode='full')
			bTot = signal.convolve(bTot, bNotch100, mode='full')
			aTot = signal.convolve(aTot, aNotch100, mode='full')
			aTot = signal.convolve(aTot, aNotch100, mode='full')
		else:

			f25 = 25.0
			w25 = f25/(glb.fs/2)
			bNotch25, aNotch25 = signal.iirnotch(w25, Q)

			bTot = signal.convolve(bTot, bNotch25, mode='full')
			bTot = signal.convolve(bTot, bNotch25, mode='full')
			aTot = signal.convolve(aTot, aNotch25, mode='full')
			aTot = signal.convolve(aTot, aNotch25, mode='full')
	elif filtertype == "lowpass":
		N  = 4    # Filter order
		fk = 48.0
		Wn = fk/(glb/2) # Cutoff frequency
		lowpassB, lowpassA = signal.butter(N, Wn, output='ba')

		bTot = signal.convolve(b, lowpassB, mode='full')
		aTot = signal.convolve(a, lowpassA, mode='full')	
	
	glb.window = len(bTot)
	glb.Zi = np.zeros([numCh,glb.window-1])

	return bTot, aTot

def plotfilter(data, b=0, a=0):

	if b[0] == 0:
		b, a = designfilter()

	Zi = signal.lfilter_zi(b, a) * data[0]
	data, Zi = signal.lfilter(b, a, data, zi=Zi)
	data = signal.lfilter(b, a, data)
	return data


def analyze_filter(Q=50):
	b, a = designfilter(filtertype = "notch", Q=Q)
	plot.plot_filterz(b, a)
	plt.subplots_adjust(hspace=0.5, wspace = 0.3)
	plt.show()

def loadfilter():
	b=np.loadtxt('bcoeff.out')
	a=np.loadtxt('acoeff.out')
	print("Load filter coefficients")
	return b,a