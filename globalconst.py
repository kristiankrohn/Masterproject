import time as tme
from threading import Lock
from scipy import signal
import numpy as np
import os, shutil



if os.name == 'nt':
	slash = "\\"
	import win32com.client as wincl

	# range -10(slow) - 10(fast)
	rate = 5

	speakLib = wincl.Dispatch("SAPI.SpVoice")

	speakLib.Rate = rate
	speakLib.Speak(" ")
	
	#print("Running on Windows system")
elif os.name == 'posix':
	slash = "/"
	#print("Running on Linux system")
#else:
	#print("Running on " + os.name)


directioncode = ["cb", "ld", "dr", "dd", "lr", "cs", "rr", "ud", "ur", "rd"]
rawdata = 0
filterdata = 1
timestamp = 2
numCh = 8
nPlots = numCh
#nSamples = 2000
nSamples = 500

datasetFolders = [slash + "Dataset1" + slash, slash + "Dataset2" + slash]
longLength = 625
shortLength = 250 ##Sjekk denne i Adrian sitt dataset
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
code_path = os.path.dirname(os.path.realpath(__file__))
machinestate = "test200"
'''
movements = ["\\Center_blink", "\\Center_still", 
			"\\Down_direction", "\\Down_return", 
			"\\Up_direction", "\\Up_return",
			"\\Left_direction", "\\Left_return", 
			"\\Right_direction", "\\Right_return", "\\Garbage"]


movements = [slash + "Center_blink", slash + "Left_direction", slash + "Down_return", slash + "Down_direction",
			slash + "Left_return", slash + "Center_still", slash + "Right_return", slash + "Up_direction",
			slash + "Up_return", slash + "Right_direction", slash + "Garbage"]			
'''
movements = [slash + "Center_blink", slash + "Up_direction", slash + "Down_return", slash + "Right_direction",
			slash + "Left_return", slash + "Center_still", slash + "Right_return", slash + "Left_direction",
			slash + "Up_return", slash + "Down_direction", slash + "Garbage"]	


MergeDict = {0:0,   1:8,  2:8,  3:6,  4:6,  5:5,  6:4,  7:4,  8:2,  9:2}

channels = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
shortclasses = [0,1,2,3,4,5,6,7,8,9]
longclasses = [0,5,2,4,6,8]

