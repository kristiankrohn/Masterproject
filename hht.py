from pyhht.visualization import plot_imfs
import numpy as np
from pyhht import EMD
import dataset
import globalvar as glb
import matplotlib.pyplot as plt
#t = np.linspace(0, 1, 1000)
#modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
#x = modes + t
def testHHT(c):
	x,y = dataset.loadDataset("longdata.txt", filterCondition=True)
			#print(y[0])
	x,y = dataset.sortDataset(x, y, length=10, classes=[c])

	for i in range(len(x[0])):
		xt = x[5][i]
		length = len(xt)
		xt = np.array(xt)
		t = np.arange(0, length/glb.fs, 1.0/glb.fs)
		#t = np.linspace(0, 1, 1000)
		#print(x[0][0])
		decomposer = EMD(xt, n_imfs=10)
		imfs = decomposer.decompose()
		print("Number of imfs: %d"%len(imfs))
		plot_imfs_test(xt, imfs, t) 
		plt.show()


def plot_imfs_test(signal, imfs, time_samples=None, fignum=None):
    """Visualize decomposed signals.

    :param signal: Analyzed signal
    :param imfs: intrinsic mode functions of the signal
    :param time_samples: time instants
    :param fignum: (optional) number of the figure to display
    :type signal: array-like
    :type time_samples: array-like
    :type imfs: array-like of shape (n_imfs, length_of_signal)
    :type fignum: int
    :return: None
    :Example:

    >>> from pyhht.visualization import plot_imfs
    >>> import numpy as np
    >>> from pyhht import EMD
    >>> t = np.linspace(0, 1, 1000)
    >>> modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
    >>> x = modes + t
    >>> decomposer = EMD(x)
    >>> imfs = decomposer.decompose()
    >>> plot_imfs(x, imfs, t) #doctest: +SKIP

    .. plot:: ../../docs/examples/simple_emd.py
    """
    if time_samples is None:
        time_samples = np.arange(signal.shape[0])

    n_imfs = imfs.shape[0]

    plt.figure(num=fignum)
    axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))

    # Plot original signal
    ax = plt.subplot(n_imfs + 1, 1, 1)
    ax.plot(time_samples, signal)
    ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('Signal')
    ax.set_title('Empirical Mode Decomposition')

    # Plot the IMFs
    for i in range(n_imfs - 1):
        print(i + 2)
        ax = plt.subplot(n_imfs + 1, 1, i + 2)
        ax.plot(time_samples, imfs[i, :])
        ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
        ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                       labelbottom=False)
        ax.grid(False)
        ax.set_ylabel('imf' + str(i + 1))

    # Plot the residue
    ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
    ax.plot(time_samples, imfs[-1, :], 'r')
    ax.axis('tight')
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('res.')
