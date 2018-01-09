import Tkinter as tk
import time as tme
import numpy as np
from numpy.random import randint
from globalvar import *
#from threading import Lock
import threading
import matplotlib.pyplot as plt
from operator import sub
filelock = Lock()
from scipy import signal
import os, shutil

size = 1000
speed = 40
ballsize = 30
startButton = None
w = None
root = None
sleeping = False
startMove = tme.time()
endMode = tme.time()
z = 3
length = 500

class Alien(object):
	def __init__(self, canvas, *args, **kwargs):
		global center, right, left, up, down, startSleep, startMove, endMove, sleeping
		self.canvas = canvas
		self.id = canvas.create_oval(*args, **kwargs)
		#self.canvas.coords(self.id, [20, 260, 120, 360])
		self.vx = 0
		self.vy = 0
		center = False
		right = False
		left = False
		up = False
		down = False
		startSleep = tme.time()
		sleeping = True

	def move(self):
		global size, speed, center, right, left, up, down, startSleep, sleeping, startMove, endMove
		global timestamp
		global z
		x1, y1, x2, y2 = self.canvas.bbox(self.id)

		if not center and ((right and (x1 <= (size/2) - ballsize)) 
						or (down and (y1 <= (size/2) - ballsize)) 
						or (left and (x2 >= (size/2) + ballsize)) 
						or (up and (y2 >= (size/2) + ballsize))):
			self.vx = 0
			self.vy = 0
			center = True
			#endMove = tme.time()
			#print("Movementtime= ")
			#print(tme.time())

			cmd = 0
			if right:
				cmd = 6
				
			elif left:
				cmd = 4

			elif up:
				cmd = 8

			elif down:
				cmd = 2

			
			threadSave = threading.Thread(target=saveTempData, args=(cmd,))
			threadSave.setDaemon(True)
			threadSave.start()

			right = False
			left = False
			up = False
			down = False
			startSleep = tme.time()
			sleeping = True
			#tme.sleep(4)

		if sleeping and (tme.time() > startSleep + 2):
			cmd = 5
			endMove = tme.time()
			#print("Movementtime= ")
			#print(tme.time())
			threadSave = threading.Thread(target=saveTempData, args=(cmd,))
			threadSave.setDaemon(True)
			threadSave.start()
			#z = randint(0,4)
			if z == 3:
				z = 0
			else:
				z = z + 1

			sleeping = False
			#print(z)
			startMove = tme.time()
			if z == 0:
				self.vx = 0
				self.vy = -speed
				print("Up")
			elif z == 1:
				self.vx = speed
				self.vy = 0
				print("Right")
			elif z == 2:
				self.vx = 0
				self.vy = speed
				print("Down")
			else:
				self.vx = -speed
				self.vy = 0
				print("Left")

		if x2 > size: #Down
			self.vx = 0
			right = True
			center = False
			cmd = 3
			#threadSave = threading.Thread(target=saveTempData, args=(cmd,))
			#threadSave.setDaemon(True)
			#threadSave.start()
			tme.sleep(1)
			self.vx = -speed

		if x1 < 0: #Up
			self.vx = 0
			left = True
			center = False
			cmd = 7
			#threadSave = threading.Thread(target=saveTempData, args=(cmd,))
			#threadSave.setDaemon(True)
			#threadSave.start()
			tme.sleep(1)
			self.vx = speed

		if y2 > size: #Right
			self.vy = 0
			down = True
			center = False
			cmd = 9
			#threadSave = threading.Thread(target=saveTempData, args=(cmd,))
			#threadSave.setDaemon(True)
			#threadSave.start()
			tme.sleep(1)
			self.vy = -speed
			
		if y1 < 0: #Left
			self.vy = 0
			up = True
			center = False
			cmd = 1
			#threadSave = threading.Thread(target=saveTempData, args=(cmd,))
			#threadSave.setDaemon(True)
			#threadSave.start()
			tme.sleep(1)
			self.vy = speed

		self.canvas.move(self.id, self.vx, self.vy)


class Ball(object):
	def __init__(self, master, **kwargs):
		self.master = master
		self.canvas = tk.Canvas(self.master, width=size, height=size)
		self.canvas.pack()
		self.aliens = Alien(self.canvas, (size/2) - ballsize, (size/2) - ballsize, (size/2) + ballsize, (size/2) + ballsize, outline='white', fill='red')
		self.canvas.pack()
		self.master.after(0, self.animation)


	def animation(self):
		#for alien in self.aliens:
		self.aliens.move()
		self.master.after(12, self.animation)

	def close_window(self):
		self.destroy()

def startgui():
	global startButton, w, root
	w.pack_forget()
	startButton.pack_forget()
	Ball(root)

def guiloop():
	global startButton, w, root
	root = tk.Tk()
	root.title("Training GUI")
	w = tk.Label(root, text="Look at the red dot, press start when ready!")
	w.pack()
	startButton = tk.Button(root, text='Start', width=25, command=startgui)
	startButton.pack()
	exitButton = tk.Button(root, text='Exit', width=25, command=root.destroy)
	exitButton.pack()
	#root = tk.Tk()
	#app = App(root)
	root.mainloop()

def saveTempData(direction):
	global data, nSamples
	global timeData
	global mutex
	global length
	numCh = 2
	
	if direction != 5:
		startTime = tme.time() + 0.4
	else: 
		startTime = tme.time()
	
	ready = False

	while not ready:
		if timeData[1][-1] > startTime:
			ready = True

	with(mutex):
		temp = data
		tempTime = timeData

	if len(temp[1]) > length:

		stopindex = len(temp[1])-5

		for i in range(len(tempTime[1])-1, 0, -1):
			if tempTime[1][i]<=startTime:
				stopindex = i
				break

		stop = stopindex
		start = stop - length
		if stop > len(temp[0])-1:
			print("Index error, aborting save operation")
		else: 
			with(filelock):

				abort = False
				f = ""
				for j in range(numCh):
					good = True
					if direction == 4:
						f += "lr"
					elif direction == 1:
						f +="ld"
					elif direction == 6:
						f += "rr"
					elif direction == 9:
						f += "rd"
					elif direction == 8:
						f += "ur"
					elif direction == 7:
						f += "ud"
					elif direction == 2:
						f += "dr"
					elif direction == 3:
						f += "dd"
					elif direction == 5:
						f += "cr"
					else:
						good = False

					if abort:
						break

					if good:
						f += str(j)
						for i in range(start, stop):
							f += ","
							try:
								num = temp[j][i]
							except IndexError:
								print("Index error")
								print("Len buffer = %d" % len(temp[j])-1)
								print("Index = %d" % i)
								print("Write operation aborted")
								abort = True
								break

							f += str(num)

						f += ":"

				if not abort:		
					file = open('temp.txt', 'a')
					file.write(f)
					file.close()



def openFile():
	global length

	dir_path = os.path.dirname(os.path.realpath(__file__))
	folder = dir_path + "\\figures\\Center"
	print(folder)
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
        	#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)
			return
	folder = dir_path + "\\figures\\Down"
	print(folder)
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
        	#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)
			return
	folder = dir_path + "\\figures\\Left"
	print(folder)
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
        	#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)
			return
	folder = dir_path + "\\figures\\Right"
	print(folder)
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
        	#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)
			return
	folder = dir_path + "\\figures\\Up"
	print(folder)
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
        	#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)
			return

	file = open('data.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	for i in range(0, len(DataSet), 4):
		care = True
		feature1 = []
		feature1 = DataSet[i].split(',')
		featuretype1 = feature1[0]
		feature1.pop(0)
		if featuretype1 == 'u0':
			title = "Up"
		elif featuretype1 == 'd0':
			title = "Down"
		elif featuretype1 == 'l0':
			title = "Left"
		elif featuretype1 == 'r0':
			title = "Right"
		elif featuretype1 == 'c0':
			title = "Center"
			#care = False
		else:
			title = featuretype1
			care = False

		if care:
			plt.suptitle(title)
			#print(title)
			#print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-100,100])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("X1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			#print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-100,100])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("X2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')


			#featureData3 = map(sub, featureData1, featureData2)
			#ax3 = plt.subplot(313)
			#ax3.set_autoscaley_on(False)
			#ax3.set_ylim([-100,100])
			#ax3.set_title("Fp1 - Fp2")
			#plt.plot(x, featureData3, label='Fp1 - Fp2')
			#plt.ylabel('uV')
			#plt.xlabel('Seconds')

			#featureData3 = map(sub, featureData1, featureData2)
			#ax3 = plt.subplot(313)
			#ax3.set_autoscaley_on(False)
			#ax3.set_xlim([1,20])
			#ax3.set_title("Fp1 - Fp2")
			#Y = abs(np.fft.fft(featureData1)/len(featureData1))
			
			#plt.plot(Y, label='Fp1 - Fp2')
			#plt.ylabel('uV')
			#plt.xlabel('Hz')
			#print(signal.hilbert(featureData1))
			savestring = "figures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			plt.savefig(savestring, bbox_inches='tight')
			#plt.show()
			plt.close()
	#plt.close('all')
	print("Finished exporting plots")

def opentemp():
	global length

	dir_path = os.path.dirname(os.path.realpath(__file__))
	folder = dir_path + "\\tempfigures\\Center"
	print(folder)
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
        	#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)
			return
	folder = dir_path + "\\tempfigures\\Down"
	print(folder)
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
        	#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)
			return
	folder = dir_path + "\\tempfigures\\Left"
	print(folder)
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
        	#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)
			return
	folder = dir_path + "\\tempfigures\\Right"
	print(folder)
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
        	#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)
			return
	folder = dir_path + "\\tempfigures\\Up"
	print(folder)
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
        	#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)
			return

	file = open('temp.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	for i in range(0, len(DataSet), 2):
		care = True
		feature1 = []
		feature1 = DataSet[i].split(',')
		featuretype1 = feature1[0]
		feature1.pop(0)
		if featuretype1 == 'ur0':
			title = "Up"
		elif featuretype1 == 'dr0':
			title = "Down"
		elif featuretype1 == 'lr0':
			title = "Left"
		elif featuretype1 == 'rr0':
			title = "Right"
		elif featuretype1 == 'cr0':
			title = "Center"
			#care = False
		else:
			title = featuretype1
			care = False

		if care:
			plt.suptitle(title)
			#print(title)
			#print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-200,200])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("X1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			#print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-200,200])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("X2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			savestring = "tempfigures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			plt.savefig(savestring, bbox_inches='tight')
			#plt.show()
			plt.close()
	plt.close('all')
	print("Finished exporting plots")
def viewdataelement(index):
	global length
	file = open('data.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	i = index * 2
	if i > len(DataSet)-2:
		print("Value is too big")
	else:
		care = True
		feature1 = []
		feature1 = DataSet[i].split(',')
		featuretype1 = feature1[0]
		feature1.pop(0)
		if featuretype1 == 'u0':
			title = "Up"
		elif featuretype1 == 'd0':
			title = "Down"
		elif featuretype1 == 'l0':
			title = "Left"
		elif featuretype1 == 'r0':
			title = "Right"
		elif featuretype1 == 'c0':
			title = "Center"
			#care = False
		else:
			title = featuretype1
			care = False

		if care:
			plt.suptitle(title)
			print(title)
			print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-100,100])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("Fp1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-100,100])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("Fp2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			savestring = "figures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			#plt.savefig(savestring, bbox_inches='tight')
			plt.show()
			#plt.close()

def viewtempelement(index):
	global length
	file = open('temp.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	i = index * 2
	if i > len(DataSet)-2:
		print("Value is too big")
	else:
		care = True
		feature1 = []
		feature1 = DataSet[i].split(',')
		featuretype1 = feature1[0]
		feature1.pop(0)
		if featuretype1 == 'u0':
			title = "Up"
		elif featuretype1 == 'd0':
			title = "Down"
		elif featuretype1 == 'l0':
			title = "Left"
		elif featuretype1 == 'r0':
			title = "Right"
		elif featuretype1 == 'c0':
			title = "Center"
			#care = False
		else:
			title = featuretype1
			care = False

		if care:
			plt.suptitle(title)
			print(title)
			print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-100,100])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("Fp1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-100,100])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("Fp2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			savestring = "figures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			#plt.savefig(savestring, bbox_inches='tight')
			plt.show()
			#plt.close()
def deletedataelement(index):
	global length
	index = index * 2
	file = open('data.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	DataSet.pop(index)
	DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open('data.txt', 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / 2
	print("Data element %d is deleted" % index)

def deletetempelement(index):
	global length
	index = index * 2
	file = open('temp.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	DataSet.pop(index)
	DataSet.pop(index)
	#print(DataSet)
	file.close()
	file = open('temp.txt', 'w')
	for i in range(len(DataSet)-1):
		file.write(DataSet[i])
		file.write(':')
	file.close()
	index = index / 2
	print("Temp element %d is deleted" % index)

def parametrization(map1, map2):
	map3 = map1 - map2


def saveData():
	tempfile = open('temp.txt', 'r')
	tempData = tempfile.read()
	tempfile.close()
	permfile = open('data.txt', 'a')
	permfile.write(tempData)
	permfile.close()
	tempfile = open('temp.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Saved")

def clearTemp():
	tempfile = open('temp.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Temp is cleared")

def clearData():
	tempfile = open('data.txt', 'w')
	tempfile.truncate(0)
	tempfile.close()
	print("Data is deleted")