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

notchZi = np.zeros([numCh,2])
notchZi2 = np.zeros([numCh,2])
notchZi3 = np.zeros([numCh,2])
notchZi4 = np.zeros([numCh,2])
notchZi1001 = np.zeros([numCh,2])
notchZi1002 = np.zeros([numCh,2])
#print(notchB)
#print(notchA)
#Butterworth lowpass filter
N  = 4    # Filter order
fk = 30.0
Wn = fk/(fs/2) # Cutoff frequency
lowpassB, lowpassA = signal.butter(N, Wn, output='ba')
lowpassZi = np.zeros([numCh,N])
DCnotchZi = np.zeros([numCh,1])
DCnotchZi2 = np.zeros([numCh,1])

#FIR bandpass filter
hcc = 56.0/(fs/2) #highest cut, only used if multibandpass
hc = 45.0/(fs/2) #High cut
lc = 5.0/(fs/2)	#Low cut

bandpassB = signal.firwin(window, hc, pass_zero=True, window = 'hann') #Bandpass
#bandpassB = signal.firwin(window, [lc, hc], pass_zero=False, window = 'hann') #Bandpass
bandpassA = 1.0 #np.ones(len(bandpassA))
bandpassZi = np.zeros([numCh, window-1])
#print(bandpassB)
highpassB = signal.firwin(window, lc, pass_zero=False, window = 'hann') #Bandpass
print("Filtersetup finished")


			

def init_filter():
	global lowpassB, lowpassA, lowpassZi 
	global bandpassB, bandpassA, bandpassZi
	global notchB, notchA, notchZi, notchZi2, notchZi3, notchZi4 
	global newSamples, nPlots
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


def filter(newSamples, newTimeData, i):
	global lowpassB, lowpassA, lowpassZi 
	global bandpassB, bandpassA, bandpassZi
	global notchB, notchA, notchZi, notchZi2, notchZi3 
	global data, window, nPLots, init, initNotch, initLowpass, initBandpass
	global bandstopFilter, lowpassFilter, bandpassFilter
	global timeData, mutex
	global DcNotchB, DcNotchA, DCnotchZi, DCnotchZi2
	global bNotch100, aNotch100, notchZi1001, notchZi1002



	x = newSamples[i]
	xr = x
	xt = newTimeData[i]
	
	x, DCnotchZi[i] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi[i])
	#x, DCnotchZi2[i] = signal.lfilter(DcNotchB,DcNotchA,x,zi=DCnotchZi2[i])

	if bandstopFilter: 
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

	
	for j in range(len(x)):

		data[i][rawdata].append(xr[j])
		data[i][filterdata].append(x[j])
		data[i][timestamp].append(xt[j])





def viewdataelement(index, filename): #Utdatert
	global length, numCh
	file = open(("Dataset\\"+filename), 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	i = index * numCh
	if i > len(DataSet)-2:
		print("Value is too big")
	else:
		care = True
		feature1 = []
		feature1 = DataSet[i].split(',')
		featuretype1 = feature1[0]
		feature1.pop(0)
		if featuretype1 == 'u0':
			title = "Up"
		elif featuretype1 == 'd0':
			title = "Down"
		elif featuretype1 == 'l0':
			title = "Left"
		elif featuretype1 == 'r0':
			title = "Right"
		elif featuretype1 == 'c0':
			title = "Center"
			#care = False
		else:
			title = featuretype1
			care = False

		if care:
			plt.suptitle(title)
			print(title)
			print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-100,100])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("Fp1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-100,100])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("Fp2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			savestring = "Dataset_exports/figures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			#plt.savefig(savestring, bbox_inches='tight')
			plt.show()
			#plt.close()

def viewtempelement(index): #Utdatert
	global length
	file = open('Dataset\\temp.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	i = index * 2
	if i > len(DataSet)-2:
		print("Value is too big")
	else:
		care = True
		feature1 = []
		feature1 = DataSet[i].split(',')
		featuretype1 = feature1[0]
		feature1.pop(0)
		if featuretype1 == 'u0':
			title = "Up"
		elif featuretype1 == 'd0':
			title = "Down"
		elif featuretype1 == 'l0':
			title = "Left"
		elif featuretype1 == 'r0':
			title = "Right"
		elif featuretype1 == 'c0':
			title = "Center"
			#care = False
		else:
			title = featuretype1
			care = False

		if care:
			plt.suptitle(title)
			print(title)
			print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-100,100])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("Fp1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-100,100])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("Fp2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			savestring = "figures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			#plt.savefig(savestring, bbox_inches='tight')
			plt.show()
			#plt.close()