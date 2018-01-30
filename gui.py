import Tkinter as tk
import time as tme
import numpy as np
from numpy.random import randint
from globalvar import *
#from threading import Lock
import threading
import matplotlib.pyplot as plt
from scipy import signal
import os, shutil
import dataset



size = 1000
speed = 40
ballsize = 30
startButton = None
wait = False
startWait = tme.time()
removeBall = False
startSleep = tme.time()
w = None
root = None
sleeping = False
startMove = tme.time()
endMode = tme.time()
z = 3


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
		global removeBall, startRemoveBall, wait, startWait
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

			
			threadSave = threading.Thread(target=dataset.saveLongTemp, args=(cmd,))
			#threadSave.setDaemon(True)
			threadSave.start()
			threadShortSave = threading.Thread(target=dataset.saveShortTemp, args=(cmd,))
			#threadShortSave.setDaemon(True)
			threadShortSave.start()
			right = False
			left = False
			up = False
			down = False
			wait=True
			startWait = tme.time()
		if wait and (tme.time()>startWait+2):
			wait=False
			removeBall = True
			startRemoveBall = tme.time()
			self.id = self.canvas.create_oval((size/2) - ballsize, (size/2) - ballsize, 
				(size/2) + ballsize, (size/2) + ballsize, 
				outline='white', fill='white')

		if removeBall and (tme.time()>startRemoveBall+0.5):
			removeBall = False
			self.id = self.canvas.create_oval((size/2) - ballsize, (size/2) - ballsize, 
				(size/2) + ballsize, (size/2) + ballsize, 
				outline='white', fill='red')
			cmd = 0
			threadSave = threading.Thread(target=dataset.saveLongTemp, args=(cmd,))
			#threadSave.setDaemon(True)
			threadSave.start()
			threadShortSave = threading.Thread(target=dataset.saveShortTemp, args=(cmd,))
			#threadShortSave.setDaemon(True)
			threadShortSave.start()


			startSleep = tme.time()
			sleeping = True
			#tme.sleep(4)

		if sleeping and (tme.time() > startSleep + 3):
			cmd = 5
			endMove = tme.time()
			#print("Movementtime= ")
			#print(tme.time())
			threadSave = threading.Thread(target=dataset.saveLongTemp, args=(cmd,))
			#threadSave.setDaemon(True)
			threadSave.start()
			threadShortSave = threading.Thread(target=dataset.saveShortTemp, args=(cmd,))
			#threadShortSave.setDaemon(True)
			threadShortSave.start()
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
			threadSave = threading.Thread(target=dataset.saveShortTemp, args=(cmd,))
			#threadSave.setDaemon(True)
			threadSave.start()
			tme.sleep(1)
			self.vx = -speed

		if x1 < 0: #Up
			self.vx = 0
			left = True
			center = False
			cmd = 7
			threadSave = threading.Thread(target=dataset.saveShortTemp, args=(cmd,))
			#threadSave.setDaemon(True)
			threadSave.start()
			tme.sleep(1)
			self.vx = speed

		if y2 > size: #Right
			self.vy = 0
			down = True
			center = False
			cmd = 9
			threadSave = threading.Thread(target=dataset.saveShortTemp, args=(cmd,))
			#threadSave.setDaemon(True)
			threadSave.start()
			tme.sleep(1)
			self.vy = -speed
			
		if y1 < 0: #Left
			self.vy = 0
			up = True
			center = False
			cmd = 1
			threadSave = threading.Thread(target=dataset.saveShortTemp, args=(cmd,))
			#threadSave.setDaemon(True)
			threadSave.start()
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

