import time as tme
from threading import Lock
from scipy import signal
import numpy as np
global rawdata
global data
global nSamples
global timeData
global newTimeData
global newSamples
global timestamp
global avgLength
global mutex
import os, shutil

#rawdata = [],[],[],[],[],[],[],[]
#data = [],[],[],[],[],[],[],[]
#timeData = [],[],[],[],[],[],[],[]

newTimeData = [],[],[],[],[],[],[],[],[]
newSamples = [],[],[],[],[],[],[],[],[]
directioncode = ["cb", "ld", "dr", "dd", "lr", "cs", "rr", "ud", "ur", "rd"]
rawdata = 0
filterdata = 1
timestamp = 2
numCh = 8
#nSamples = 1000
nSamples = 2000
#avgLength = 1000
avgLength = 25
avgShortLength = 50
xt = tme.time()
fs = 250
mutex = Lock()
nPlots = numCh
window = 3
init = True
bandstopFilter = True
lowpassFilter = False
bandpassFilter = False
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

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
		
#print(os.listdir(dir_path))  


