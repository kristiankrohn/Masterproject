import time as tme
from threading import Lock
from scipy import signal
import numpy as np
import os, shutil
from globalconst import*
#rawdata = [],[],[],[],[],[],[],[]
#data = [],[],[],[],[],[],[],[]
#timeData = [],[],[],[],[],[],[],[]

newTimeData = [],[],[],[],[],[],[],[],[]
newSamples = [],[],[],[],[],[],[],[],[]




xt = tme.time()
fs = 250.0
b = None
a = None
window = 10
Zi = np.zeros([numCh,window-1])
mutex = Lock()

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


