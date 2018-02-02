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
datasetFolders = ["\\Dataset", "\\Dataset_Aexternal"]

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

movements = ["\\Center_blink", "\\Center_still", 
			"\\Down_direction", "\\Down_return", 
			"\\Up_direction", "\\Up_return",
			"\\Left_direction", "\\Left_return", 
			"\\Right_direction", "\\Right_return", "\\Garbage"]

channels = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]