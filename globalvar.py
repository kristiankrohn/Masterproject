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
#rawdata = [],[],[],[],[],[],[],[]
#data = [],[],[],[],[],[],[],[]
data = [[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]],
		[[],[],[]]]

#timeData = [],[],[],[],[],[],[],[]
newTimeData = [],[],[],[],[],[],[],[]
newSamples = [],[],[],[],[],[],[],[]

rawdata = 0
filterdata = 1
timestamp = 2

#nSamples = 1000
nSamples = 1800
#avgLength = 1000
avgLength = 25
avgShortLength = 50
timestamp = tme.time()
fs = 250
mutex = Lock()
nPlots = 8
window = 3
init = True
bandstopFilter = True
lowpassFilter = False
bandpassFilter = False

#data.append(Data(rawdata, filterdata, timestamp))
#class rtData():
	#"Container for data"
	#def __init__(self, rawdata, filterdata, timestamp):
		#self.rawdata = rawdata
		#self.filterdata = filterdata
		#self.timestamp = timestamp
	
#class newData():
	#"Container for new data before processing"
	#def __init__(self, rawdata, timestamp):
		#self.rawdata = rawdata
		#self.timestamp = timestamp



