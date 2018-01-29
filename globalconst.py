import time as tme
from threading import Lock
from scipy import signal
import numpy as np
import os, shutil

directioncode = ["cb", "ld", "dr", "dd", "lr", "cs", "rr", "ud", "ur", "rd"]
rawdata = 0
filterdata = 1
timestamp = 2
numCh = 9
nPlots = numCh
nSamples = 2000

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))