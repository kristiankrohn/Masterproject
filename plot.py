from globalvar import *

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