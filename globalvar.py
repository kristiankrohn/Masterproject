import time as tme
from threading import Lock
from scipy import signal
import numpy as np
import os, shutil
from globalconst import*
#rawdata = [],[],[],[],[],[],[],[]
#data = [],[],[],[],[],[],[],[]
#timeData = [],[],[],[],[],[],[],[]
import getpass

newTimeData = [],[],[],[],[],[],[],[],[]
newSamples = [],[],[],[],[],[],[],[],[]




xt = tme.time()
fs = 250.0
b = None
a = None
window = 10
Zi = np.zeros([numCh,window-1])
mutex = Lock()

if os.name == 'nt':
	if getpass.getuser() == "Kristian":
		datasetFolder = datasetFolders[0]
	else:
		datasetFolder = datasetFolders[1]
	#print("Running on Windows system")
elif os.name == 'posix':
	datasetFolder = datasetFolders[0]
	#print("Running on Linux system")
else:
	print("Running on " + os.name)
	datasetFolder = datasetFolders[0]

#data[channel][filterdata][sample]
guipredict = True
saveData = False
oldSampleID = 255
if numCh == 8:
	data = [[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]]]
elif numCh == 9:
	data = [[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]]]


		
#print(os.listdir(dir_path))  


