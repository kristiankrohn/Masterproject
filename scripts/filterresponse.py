from scipy import signal
from scipy.signal import kaiserord, firwin
import numpy as np
import matplotlib.pyplot as plt

#window = 75
fs = 250.0
f0 = 50.0
Q = 50
w0 = f0/(fs/2)
bNotch, aNotch = signal.iirnotch(w0, Q)

f100 = 100.0
w100 = f100/(fs/2)
bNotch100, aNotch100 = signal.iirnotch(w100, Q)
print(bNotch100)
print(aNotch100)
nyq_rate = fs / 2.0
N = 151

# The cutoff frequency of the filter.
cutoff_hz = 3.0

# Use firwin with a Kaiser window to create a lowpass FIR filter.
#a = firwin(N, cutoff_hz/nyq_rate, pass_zero=False, window = 'hann')



#Bandpass
#bBand = firwin(N, [cutoff_hz/nyq_rate, 44/nyq_rate, 56/nyq_rate], pass_zero=False, window = 'hann')
bHigh = firwin(N, cutoff_hz/nyq_rate, pass_zero=False, window = 'hann')
bBand = firwin(N, [cutoff_hz/nyq_rate, 44/nyq_rate], pass_zero=False, window = 'hann')
bFir = firwin(N, 45/nyq_rate, pass_zero=True, window = 'hann')
b1000 = np.repeat(1.0/1000, 1000)
b1000 = -b1000
b1000[0] = b1000[0] + 1

b500 = np.repeat(1.0/500, 500)
b500 = -b500
b500[0] = b500[0] + 1

b250 = np.repeat(1.0/250, 250)
b250 = -b250
b250[0] = b250[0] + 1

b100 = np.repeat(1.0/100, 100)
b100 = -b100
b100[0] = b100[0] + 1

b10 = np.repeat(1.0/10, 10)
b10 = -b10
b10[0] = b10[0] + 1

b5 = np.repeat(1.0/5, 5)
b5 = -b5
b5[0] = b5[0] + 1

bLong = np.repeat(1.0/50, 50)
bShort = np.repeat(1.0/25, 25)
bLong = -bLong
bShort = - bShort
bLong[0] = bLong[0] + 1
bShort[0] = bShort[0] + 1 
#bShort.resize(bLong.shape)
#bShortOne = bShort
#bDual = signal.convolve(bLong, bShort, mode='full')
#for i in range(len(bShortOne)):
	#if bShortOne[i] == 0:
		#bShortOne[i] = 1
#print(bShort)
N  = 30    # Filter order
fk = 45
Wn = fk/(fs/2) # Cutoff frequency
#b, a = signal.butter(N, Wn, output='ba')


#a = [1 , -0.9,0.0000000000001] 
#b = [1,-1,0.000000000000001]

a = [1 , -0.9] 
b = [1,-1]


#a = [1 , -2,1] 
#b = [1,-2,1]
#b, a = signal.iirfilter(1, 0, btype='bandstop')
print(b)
print(a)
#bHat = signal.convolve(b, b, mode='full')
#aHat = signal.convolve(a, a, mode='full')
#print(xHat)
#bTot = signal.convolve(1, -bShort)

average = False
dc50notch = False
dualaverage = False
highpass = False
bandpass = False
septanotch = True


xD50 = signal.convolve(bShort, bShort, mode='full')
bD50 = signal.convolve(xD50, bNotch, mode='full')
bD50 = signal.convolve(bD50, bNotch, mode='full')
aD50 = signal.convolve(aNotch, aNotch, mode='full')

xD100 = signal.convolve(bLong, bLong, mode='full')
bD100 = signal.convolve(xD100, bNotch, mode='full')
bD100 = signal.convolve(bD100, bNotch, mode='full')
aD100 = signal.convolve(aNotch, aNotch, mode='full')

if average:
	bTot = bLong
	aTot = 1

elif dc50notch:
	bTot = signal.convolve(b, bNotch, mode='full')
	aTot = signal.convolve(a, aNotch, mode='full')

elif dualaverage:
	#xHat = signal.convolve(bShort, bShort, mode='full')
	xHat = signal.convolve(bShort, bLong, mode='full')
	#xHat = signal.convolve(xHat, b1000, mode='full')
	#xHat = signal.convolve(xHat, b500, mode='full')
	#xHat = signal.convolve(xHat, b250, mode='full')
	#xHat = signal.convolve(xHat, b100, mode='full')
	#xHat = signal.convolve(xHat, b10, mode='full')
	#xHat = signal.convolve(xHat, b5, mode='full')
	bTot = signal.convolve(xHat, bNotch, mode='full')
	bTot = signal.convolve(bTot, bNotch, mode='full')
	aTot = signal.convolve(aNotch, aNotch, mode='full')
	#aTot = 1

elif highpass:
	bTot = signal.convolve(bHigh, bLong, mode='full')
	bTot = signal.convolve(bTot, bNotch, mode='full')
	aTot = aNotch

elif bandpass:
	bTot = signal.convolve(bBand, bLong, mode='full')
	aTot = 1

elif septanotch:
	bTot = signal.convolve(b, bNotch, mode='full')
	bTot = signal.convolve(bTot, bNotch, mode='full')
	bTot = signal.convolve(bTot, bNotch, mode='full')
	bTot = signal.convolve(bTot, bNotch100, mode='full')
	bTot = signal.convolve(bTot, bNotch, mode='full')
	bTot = signal.convolve(bTot, bNotch100, mode='full')
	aTot = signal.convolve(a, aNotch, mode='full')
	aTot = signal.convolve(aTot, aNotch, mode='full')
	aTot = signal.convolve(aTot, aNotch, mode='full')
	aTot = signal.convolve(aTot, aNotch100, mode='full')
	aTot = signal.convolve(aTot, aNotch, mode='full')
	aTot = signal.convolve(aTot, aNotch100, mode='full')



#print(aTot)
#print(bTot)      
#bTot = bShort
#bAvg1 = (1 - bShort - bLong)
#bAvgMul = bLong * bShortOne
#bAvg = bAvg1 + bAvgMul
#bFir.resize(bAvg.shape)
#bTot = bShort
#for i in range(len(bFir)):
	#if bFir[i] == 0:
		#bFir[i] = 1
#bTot = bAvg*bFir
#bTot = 1 - bLong
#print(bTot)
#Highpass
#a = signal.firwin(N, cutoff_hz/nyq_rate, pass_zero=False, window = 'hann') #Bandpass
#a = -a
#a[n/2] = a[n/2] + 1


#order = 2
#hc = 40.0/(fs/2)
#lc = 2.0/(fs/2)
#b, a = signal.butter(order, [lc, hc], btype='band', output='ba', analog=False)
#print(b)
#print(a)


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


def plot_freqz(ax,b,a=1):
	global legends
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
	legend, = plt.plot(w*fs/(2*np.pi), h_dB, label=label)
	legends.append(legend)
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
		l = 100
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
		l = 100
	impulse = np.repeat(0.,l); impulse[0] =1.
	x = np.arange(0,l)
	response = signal.lfilter(b,a,impulse)
	step = np.cumsum(response)
	plt.plot(x, step)
	plt.ylabel('Amplitude')
	plt.xlabel(r'n (samples)')
	plt.title(r'Step response')

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

legends = []
np.savetxt('bcoeff.out', bTot)
np.savetxt('acoeff.out', aTot)
plot_filterz(bTot, aTot)
#ax = plt.subplot(211)
#plot_freqz(ax, bTot, aTot)
#plot_freqz(ax, bD50, aD50)
#plot_freqz(ax, bD100, aD100)
#plt.legend(handles=legends)
#plt.subplot(212)
#plot_stepz(bTot, aTot)
#plot_stepz(bD50, aD50)
#plot_stepz(bD100, aD100)
#plot_freqz(ax, b1000)
#plot_freqz(ax, b500)
#plot_freqz(ax, b250)
#plot_freqz(ax, b100)
#plot_freqz(ax, bLong)
#plot_freqz(ax, bShort)


#plt.legend(handles=legends)
plt.subplots_adjust(hspace=0.5, wspace = 0.3)
plt.show()
