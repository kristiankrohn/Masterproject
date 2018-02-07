from pyhht.visualization import plot_imfs
import numpy as np
from pyhht import EMD
import dataset
import globalvar as glb
#t = np.linspace(0, 1, 1000)
#modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
#x = modes + t
def testHHT(c):
	x,y = dataset.loadDataset("longtemp.txt")
			#print(y[0])
	x,y = dataset.sortDataset(x, y, length=10, classes=[c])

	for i in range(len(x[0])):
		xt = x[0][i]
		length = len(xt)
		#x = np.array(x)
		t = np.arange(0, length/glb.fs, 1.0/glb.fs)
		#t = np.linspace(0, 1, 1000)
		#print(x[0][0])
		decomposer = EMD(xt, n_imfs=5)
		imfs = decomposer.decompose()
		print("Number of imfs: %d"%len(imfs))
		plot_imfs(xt, imfs, t) 