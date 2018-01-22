from globalvar import *
import dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import blackman
import scipy.fftpack


def plot():
	global mutex
	with(mutex):
		while len(data[0][filterdata]) > nSamples:
			for i in range(nPlots):
				#data[i].pop(0)
				dataset.databufferPop()
		x = np.arange(0, len(data[1][filterdata])/fs, 1/fs)
		legends = []
		for i in range(2):
			label = "Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			legend, = plt.plot(x, data[i][filterdata], label=label)
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
		legend, = plt.plot(data[i][filterdata], label=label)
		legends.append(legend)
	plt.ylabel('uV')
	plt.xlabel('Sample')
	plt.legend(handles=legends)
	legends = []
	plt.show()

def fftplot(channel, title=""):
	global fs
	y = data[channel][filterdata]
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

def exportplot(plotdata,  title="", ax=None):
	if ax == None:
		fig, ax = plt.subplots()

	length = len(plotdata)
	x = np.arange(0, length/250.0, 1.0/250.0)
	ax.set_autoscaley_on(False)
	ax.set_ylim([-100,100])
	plt.plot(x, plotdata, label=title)
	ax.set_title(title)
	plt.ylabel('uV')
	plt.xlabel('Seconds')

def exportRaw(plotdata,  title="", ax=None):
	if ax == None:
		fig, ax = plt.subplots()

	length = len(plotdata)
	x = np.arange(0, length/250.0, 1.0/250.0)
	#ax.set_autoscaley_on(False)
	#ax.set_ylim([-100,100])
	plt.plot(x, plotdata, label=title)
	ax.set_title(title)
	plt.ylabel('uV')
	plt.xlabel('Seconds')

def exportFftPlot(plotdata, title="", ax=None):
	global fs

	# Number of samplepoints
	N = len(plotdata)
	# sample spacing
	T = 1.0 / fs
	x = np.linspace(0.0, N*T, N)
	
	w = blackman(N)
	yf = scipy.fftpack.fft(plotdata*w)
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

def plot_filterz(b, a=1):
	ax=plt.subplot(221)
	plot_freqz(ax, b, a)
	ax=plt.subplot(222)
	plot_phasez(ax, b, a)
	plt.subplot(223)
	plot_impz(b, a)
	plt.subplot(224)
	plot_stepz(b, a)
	plt.subplots_adjust(hspace=0.5, wspace = 0.3)

def plot_freqz(ax,b,a=1):
	#global legends
	w,h = signal.freqz(b,a)
	h_dB = 20 * np.log10 (abs(h))
	if len(b) == 78:
		label = "N = 50 + 25"
	elif len(b) == 53:
		label = "N = 25 + 25"
	elif len(b) == 103:
		label = "N = 50 + 50"
	else:
		label = "N = %d" %(len(b))
	#legend, = plt.plot(w*fs/(2*np.pi), h_dB, label=label)
	plt.plot(w*fs/(2*np.pi), h_dB)
	#legends.append(legend)
	#plt.plot(w/np.pi, h_dB)
	ax.set_xlim([-1, 125])
	plt.ylim([max(min(h_dB), -100) , 5])
	plt.ylabel('Magnitude (db)')
	plt.xlabel("Frequency (Hz)")
	#plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
	plt.title(r'Amplitude response')

def plot_phasez(ax, b, a=1):
	w,h = signal.freqz(b,a)
	#h_Phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))
	h_Phase = np.unwrap(np.angle(h))*180/np.pi
	plt.plot(w*fs/(2*np.pi), h_Phase)
	#plt.plot(w/np.pi, h_Phase)
	ax.set_xlim([-1, 125])
	plt.ylabel('Phase (Degrees)')
	plt.xlabel("Frequency (Hz)")
	#plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
	plt.title(r'Phase response')

def plot_impz(b, a = 1):
	if type(a)== int: #FIR
		l = len(b)
	else: # IIR
		l = 250
	impulse = np.repeat(0.,l); impulse[0] =1.
	x = np.arange(0,l)
	response = signal.lfilter(b, a, impulse)
	#plt.stem(x, response, linefmt='b-', basefmt='b-', markerfmt='bo')
	plt.plot(x, response)
	plt.ylabel('Amplitude')
	plt.xlabel(r'n (samples)')
	plt.title(r'Impulse response')

def plot_stepz(b, a = 1):
	if type(a)== int: #FIR
		l = len(b)
	else: # IIR
		l = 250
	impulse = np.repeat(0.,l); impulse[0] =1.
	x = np.arange(0,l)
	response = signal.lfilter(b,a,impulse)
	step = np.cumsum(response)
	plt.plot(x, step)
	plt.ylabel('Amplitude')
	plt.xlabel(r'n (samples)')
	plt.title(r'Step response')

def ampandphase(b, a = 1):
	w, h = signal.freqz(b, a)
	 # Generate frequency axis
	freq = w*fs/(2*np.pi)
	 # Plot
	fig, ax = plt.subplots(2, 1, figsize=(8, 6))
	ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
	ax[0].set_title("2 Step Average + Lowpass")
	ax[0].set_ylabel("Amplitude (dB)", color='blue')
	ax[0].set_xlim([0, 1])
	ax[0].set_ylim([-100, 10])
	ax[0].grid()
	ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
	ax[1].set_ylabel("Angle (degrees)", color='green')
	ax[1].set_xlabel("Frequency (Hz)")
	ax[1].set_xlim([0, 100])
	ax[1].set_yticks([-7560, -6480, -5400, -4320, -3240, -2160, -1080, 0, 1080])
	ax[1].set_ylim([-8000, 1080])
	ax[1].grid()
	plt.show()

