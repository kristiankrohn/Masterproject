from globalconst import*
import globalvar as glb
import dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import blackman
import scipy.fftpack
from sklearn.model_selection import learning_curve



def plot():
	global mutex
	with(mutex):
		while len(glb.data[0][filterdata]) > nSamples:
			for i in range(nPlots):
				dataset.databufferPop()
		x = np.arange(0, len(glb.data[1][filterdata])/glb.fs, 1/glb.fs)
		legends = []
		for i in range(2):
			label = "Fp %d" %(i+1)
			#print(label)
			#label = tuple([label])
			legend, = plt.plot(x, glb.data[i][filterdata], label=label)
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
		legend, = plt.plot(glb.data[i][filterdata], label=label)
		legends.append(legend)
	plt.ylabel('uV')
	plt.xlabel('Sample')
	plt.legend(handles=legends)
	legends = []
	plt.show()

def fftplot(channel, title=""):
	#global fs
	y = glb.data[channel][filterdata]
	# Number of samplepoints
	N = len(y)
	print(glb.fs)
	T = 1.0 / glb.fs
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
	x = np.arange(0, length/glb.fs, 1.0/glb.fs)
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
	x = np.arange(0, length/glb.fs, 1.0/glb.fs)
	#ax.set_autoscaley_on(False)
	#ax.set_ylim([-100,100])
	plt.plot(x, plotdata, label=title)
	ax.set_title(title)
	plt.ylabel('uV')
	plt.xlabel('Seconds')

def exportFftPlot(plotdata, title="", ax=None):

	# Number of samplepoints
	N = len(plotdata)
	# sample spacing
	T = 1.0 / glb.fs
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
	#legend, = plt.plot(w*glb.fs/(2*np.pi), h_dB, label=label)
	plt.plot(w*glb.fs/(2*np.pi), h_dB)
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
	plt.plot(w*glb.fs/(2*np.pi), h_Phase)
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
	if type(a) == int: #FIR
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
	freq = w*glb.fs/(2*np.pi)
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

def trainingPredictions(clf, X, y):
	from sklearn.decomposition import TruncatedSVD
	from sklearn.decomposition import PCA
	from sklearn import svm
	#X = TruncatedSVD().fit_transform(X)
	X = PCA(n_components = 2).fit_transform(X)
	#fig = plt.figure(figsize=(9, 8))
	#ax = plt.subplot(221)
	#ax.scatter(X[:, 10], X[:, 2], c=y, s=50, edgecolor='k')
	#ax.set_title("Original Data between two features(2d)")
	#ax.set_xticks(())
	#ax.set_yticks(())

	#ax = plt.subplot(222)
	#ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=50, edgecolor='k')
	#ax.set_title("Truncated SVD reduction (2d) of transformed data (d)")
	#ax.set_xticks(())
	#ax.set_yticks(())
	C = 50
	models = (svm.SVC(kernel = 'rbf', gamma = 0.01, decision_function_shape = 'ovr'),
	      svm.LinearSVC(C=C),
	      svm.SVC(kernel='linear', C=C))
	models = (clf.fit(X, y) for clf in models)

	# title for the plots
	titles = ('SVC with RBF kernel',
	      'LinearSVC (linear kernel)',
	      'SVC with linear kernel',)

	# Set-up 2x2 grid for plotting.
	fig, sub = plt.subplots(3, 1)
	plt.subplots_adjust(wspace=0.4, hspace=0.4)

	X0, X1 = X[:, 0], X[:, 1] #two first features
	xx, yy = make_meshgrid(X0, X1)

	for clf, title, ax in zip(models, titles, sub.flatten()):
	    plot_contours(ax, clf, xx, yy,
	                  cmap=plt.cm.coolwarm, alpha=0.8)
	    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
	    ax.set_xlim(xx.min(), xx.max())
	    ax.set_ylim(yy.min(), yy.max())
	    ax.set_xlabel('Feature 1')
	    ax.set_ylabel('Feature 2')
	    ax.set_xticks(())
	    ax.set_yticks(())
	    ax.set_title(title)
	    plt.plot(label = y)

	plt.legend()
	plt.show()

def make_meshgrid(x, y, h=.02):
	"""Create a mesh of points to plot in

	Parameters
	----------
	x: data to base x-axis meshgrid on
	y: data to base y-axis meshgrid on
	h: stepsize for meshgrid, optional

	Returns
	-------
	xx, yy : ndarray
	"""
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))
	return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
	"""Plot the decision boundaries for a classifier.

	Parameters
	----------
	ax: matplotlib axes object
	clf: a classifier
	xx: meshgrid ndarray
	yy: meshgrid ndarray
	params: dictionary of params to pass to contourf, optional
	"""
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	out = ax.contourf(xx, yy, Z, **params)
	return out

def learningCurve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.01, 1.0, 20)):

	"""
	Generate a simple plot of the test and training learning curve.

	Parameters
	----------
	estimator : object type that implements the "fit" and "predict" methods
	    An object of that type which is cloned for each validation.

	title : string
	    Title for the chart.

	X : array-like, shape (n_samples, n_features)
	    Training vector, where n_samples is the number of samples and
	    n_features is the number of features.

	y : array-like, shape (n_samples) or (n_samples, n_features), optional
	    Target relative to X for classification or regression;
	    None for unsupervised learning.

	ylim : tuple, shape (ymin, ymax), optional
	    Defines minimum and maximum yvalues plotted.

	cv : int, cross-validation generator or an iterable, optional
	    Determines the cross-validation splitting strategy.
	    Possible inputs for cv are:
	      - None, to use the default 3-fold cross-validation,
	      - integer, to specify the number of folds.
	      - An object to be used as a cross-validation generator.
	      - An iterable yielding train/test splits.

	    For integer/None inputs, if ``y`` is binary or multiclass,
	    :class:`StratifiedKFold` used. If the estimator is not a classifier
	    or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

	    Refer :ref:`User Guide <cross_validation>` for the various
	    cross-validators that can be used here.

	n_jobs : integer, optional
	    Number of jobs to run in parallel (default 1).
	"""

	plt.figure()
	plt.title(title)
	if ylim is not None:
	    plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(
	    estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	print(test_scores_mean)
	print(test_scores_std)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.1,
	                 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
	         label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
	         label="Cross-validation score")

	plt.legend(loc="best")
	return plt
