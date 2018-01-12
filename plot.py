from globalvar import *
import dataset
import numpy as np
import matplotlib.pyplot as plt

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
