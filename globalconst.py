import time as tme
from threading import Lock
from scipy import signal
import numpy as np
import os, shutil

directioncode = ["cb", "ld", "dr", "dd", "lr", "cs", "rr", "ud", "ur", "rd"]
rawdata = 0
filterdata = 1
timestamp = 2
numCh = 8
nPlots = numCh
nSamples = 2000
datasetFolders = ["\\Dataset\\", "\\Dataset_Aexternal\\"]
longLength = 625
shortLength = 250
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
machinestate = "learningState"
movements = ["\\Center_blink", "\\Center_still", 
			"\\Down_direction", "\\Down_return", 
			"\\Up_direction", "\\Up_return",
			"\\Left_direction", "\\Left_return", 
			"\\Right_direction", "\\Right_return", "\\Garbage"]

channels = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
shortclasses = [0,1,2,3,4,5,6,7,8,9]
longclasses = [0,5,2,4,6,8]