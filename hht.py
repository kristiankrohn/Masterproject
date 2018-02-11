from pyhht.visualization import plot_imfs
import numpy as np
from pyhht import EMD
import dataset
import globalvar as glb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from globalconst import*
#t = np.linspace(0, 1, 1000)
#modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
#x = modes + t
def testHHT(c):
	x,y = dataset.loadDataset("longdata.txt", filterCondition=True)
			#print(y[0])
	x,y = dataset.sortDataset(x, y, length=1, classes=[c])
	n_imfs = 7
	channelsToPlot = 8

	for i in range(len(x[0])): #Iternate over number of elements in dataset
		title = movements[y[0][i]]
		print title
		title = title[1:]
		print title
		fig = plt.figure(figsize=(20,10))
		#fig = plt.figure()
		plt.suptitle(title)

		outer = gridspec.GridSpec(1, channelsToPlot, wspace=0.1, hspace=0.2, right=0.98, left=0.02, bottom=0.02, top=0.95 )

		for j in range(channelsToPlot): #Iterate over channels
			inner = gridspec.GridSpecFromSubplotSpec(n_imfs + 1, 1,
                    subplot_spec=outer[j], wspace=0.1, hspace=0.1)

			xt = x[j][i]
			length = len(xt)
			xt = np.array(xt)
			t = np.arange(0, length/glb.fs, 1.0/glb.fs)
			#t = np.linspace(0, 1, 1000)
			#print(x[0][0])
			
			decomposer = EMD(xt, n_imfs=n_imfs)
			imfs = decomposer.decompose()


			time_samples = t
			signal = xt

			ax = plt.Subplot(fig, inner[0])
			#ax = plt.subplot(n_imfs + 1, 1, 1)
			ax.plot(time_samples, signal)
			ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
			ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
			   labelbottom=False)
			ax.grid(False)
			ax.set_ylabel('Signal')
			ax.set_title(channels[j])
			fig.add_subplot(ax)

			axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))
			for k in range(n_imfs - 1):
				#print(i + 2)
				ax = plt.Subplot(fig, inner[k+1])
				#ax = plt.subplot(n_imfs + 1, 1, i + 2)
				ax.plot(time_samples, imfs[k, :])
				ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
				ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
				               labelbottom=False)
				ax.grid(False)
				ax.set_ylabel('imf' + str(k + 1))
				fig.add_subplot(ax)

			ax = plt.Subplot(fig, inner[n_imfs])
			#ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
			ax.plot(time_samples, imfs[-1, :], 'r')
			ax.axis('tight')
			ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
			               labelbottom=False)
			ax.grid(False)
			ax.set_ylabel('res.')
			fig.add_subplot(ax)

		plt.show()

		#savestring = (dir_path + "\\Dataset_hht\\")
		#with filelock:
			#plt.savefig(savestring, bbox_inches='tight')
		#plt.close()