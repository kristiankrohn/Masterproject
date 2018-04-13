from globalconst import  *
import globalvar as glb
import keyboard
import time as tme
import copy
import os
import sys; sys.path.append('.') # help python find ps_drone.py relative to scripts folder
sys.path.append('../ps_drone')

def droneController():
	# Modified version of firstvideo.py 
	#########
	# firstVideo.py
	# This program is part of the online PS-Drone-API-tutorial on www.playsheep.de/drone.
	# It shows the general usage of the video-function of a Parrot AR.Drone 2.0 using the PS-Drone-API.
	# The drone will stay on the ground.
	# Dependencies: a POSIX OS, openCV2, PS-Drone-API 2.0 beta or higher.
	# (w) J. Philipp de Graaff, www.playsheep.de, 2014
	##########
	# LICENCE:
	#   Artistic License 2.0 as seen on http://opensource.org/licenses/artistic-license-2.0 (retrieved December 2014)
	#   Visit www.playsheep.de/drone or see the PS-Drone-API-documentation for an abstract from the Artistic License 2.0.
	###########


	##### Check for running posix system, exit if not #####
	if os.name == 'nt':
		print("Controller can not run on Windows system, returning")
		return

	elif os.name == 'posix':
		print("Running on Linux system, all good")
	

	##### Suggested clean drone startup sequence #####
	#import time, sys
	import ps_drone                                              # Import PS-Drone


	drone = ps_drone.Drone()                                     # Start using drone
	drone.startup()                                              # Connects to drone and starts subprocesses

	drone.reset()                                                # Sets drone's status to good (LEDs turn green when red)
	while (drone.getBattery()[0] == -1):      time.sleep(0.1)    # Waits until drone has done its reset
	print "Battery: "+str(drone.getBattery()[0])+"%  "+str(drone.getBattery()[1])	# Gives a battery-status
	drone.useDemoMode(True)                                      # Just give me 15 basic dataset per second (is default anyway)

	##### Mainprogram begin #####
	drone.setConfigAllID()                                       # Go to multiconfiguration-mode
	drone.sdVideo()
	#drone.hdVideo()                                              # Choose lower resolution (hdVideo() for...well, guess it)
	drone.frontCam()                                             # Choose front view
	CDC = drone.ConfigDataCount
	while CDC == drone.ConfigDataCount:       time.sleep(0.0001) # Wait until it is done (after resync is done)
	drone.startVideo()                                           # Start video-function
	drone.showVideo()                                            # Display the video

	drone.trim()                                       # Recalibrate sensors
	drone.getSelfRotation(5)                           # Get auto-alteration of gyroscope-sensor
	print "Auto-alt.:"+str(drone.selfRotation)+"dec/s" # Showing value for auto-alteration

	##### Controller variables ######
	translate = {0:'up', 2:'down', 4:'left', 6:'right',8:'down'}
	opposite = {2:8, 4:6, 6:4, 8:2, 0:0}
	otherkey = {2:[4,6], 4:[2,8], 6:[2,8], 8:[4,6], 0:[2,4,6,8]}
	pressedKey = None
	keypress = False
	previousPrediction = 0
	gotfive = False
	gotopposite = False
	gotother = False
	brainz = False

	##### And action !
	print "Use <space> to toggle front- and groundcamera, any other key to stop"
	IMC =    drone.VideoImageCount                               # Number of encoded videoframes
	stop =   False
	ground = False
	
	while not stop:
		while drone.VideoImageCount == IMC: tme.sleep(0.01)     # Wait until the next video-frame
		IMC = drone.VideoImageCount
		key = drone.getKey()                                     # Gets a pressed key

		if key == " ":
			if drone.NavData["demo"][0][2] and not drone.NavData["demo"][0][3]:
				drone.takeoff()
			else:																
				drone.land()

		elif key=="x":	
			if ground:              
				ground = False
			else:                   
				ground = True
			drone.groundVideo(ground)                            # Toggle between front- and groundcamera. Hint: options work for all videocommands

		elif key=="z":
			if brainz:
				brainz = False
				print("No longer flying with brainz")
			else:
				brainz = True
				print("Flying by brain, GO!")

			drone.hover()

		elif key == "0":	drone.hover()
		elif key == "w":	drone.moveForward()
		elif key == "s":	drone.moveBackward()
		elif key == "a":	drone.moveLeft()
		elif key == "d":	drone.moveRight()
		elif key == "q":	drone.turnLeft()
		elif key == "e":	drone.turnRight()
		elif key == "7":	drone.turnAngle(-10,1)
		elif key == "9":	drone.turnAngle( 10,1)
		elif key == "4":	drone.turnAngle(-45,1)
		elif key == "6":	drone.turnAngle( 45,1)
		elif key == "1":	drone.turnAngle(-90,1)
		elif key == "3":	drone.turnAngle( 90,1)
		elif key == "8":	drone.moveUp()
		elif key == "2":	drone.moveDown()
		elif key == "*":	drone.doggyHop()
		elif key == "+":	drone.doggyNod()
		elif key == "-":	drone.doggyWag()
		elif key != "":		stop = True

		elif key and key != " ":    stop =   True

		#Brain controller
		if brainz:
			if len(glb.predictions) >= 1:
				with glb.predictionslock:
					prediction = copy.copy(glb.predictions[0])
					glb.predictions.pop(0)
				if (prediction != 5) and (prediction != previousPrediction):
					if (prediction in [2,4,6,8]) and (keypress == False): #Press
						keypress = True
						pressedKey = prediction
						#Make command
						print("Press key: %d" %prediction)
						if prediction == 8:					
							drone.moveForward(0.1)
						elif prediction == 2:
							drone.moveBackward(0.1)
						elif prediction == 4:
							drone.turnAngle(-10,0.5)

						elif prediction == 6: 
							drone.turnAngle( 10,0.5)

					elif prediction == 0: #Blink
						if previousPrediction != 0:
							#Append a value to a queue 
							pass
					
				if keypress and (prediction != previousPrediction):
					if prediction == 5:
						gotfive = True

					elif prediction == opposite[pressedKey]: #Release

						gotopposite = True
						print("Got opposite key: %d" %prediction)
						drone.hover()

					elif prediction in otherkey[pressedKey]: #Release and press new

						gotother = True
						'''
						if prediction == 8:					
							drone.moveForward(0.1)
						elif prediction == 2:
							drone.moveBackward(0.1)
						elif prediction == 4:
							drone.turnLeft(0.1)
						elif prediction == 6: 
							drone.turnRight(0.1)
						'''
						drone.hover()
						
					if gotfive and ((gotopposite == True) or (gotother == True)):
						
						gotother = False
						gotopposite = False
						gotfive = False
						keypress = False
						print("Keypress = False")
					
				previousPrediction = prediction

				#If queue is longer than 3 takeoff or land

				#Delete an element from queue every 1 second

def droneSimulatorController():
	translate = {2:'down', 4:'a', 6:'d',8:'up'}
	opposite = {2:8, 4:6, 6:4, 8:2, 0:0}
	otherkey = {2:[4,6], 4:[2,8], 6:[2,8], 8:[4,6], 0:[2,4,6,8]}
	pressedKey = None
	keypress = False
	previousPrediction = 0
	gotfive = False
	gotopposite = False
	gotother = False
	brainz = False

	while True:
		if len(glb.predictions) >= 1:
			with glb.predictionslock:
				prediction = copy.copy(glb.predictions[0])
				glb.predictions.pop(0)
			if (prediction != 5) and (prediction != previousPrediction):
				if (prediction in [2,4,6,8]) and (keypress == False): #Press
					keypress = True
					pressedKey = prediction
					#Make command
					keyboard.press(translate[prediction])
					print("Press key: %d" %prediction)

				elif prediction == 0: #Blink
					if previousPrediction != 0:
						print("Got blink") 
						pass
				
			if keypress and (prediction != previousPrediction):
				if prediction == 5:
					gotfive = True

				elif prediction == opposite[pressedKey]: #Release

					gotopposite = True
					print("Got opposite key: %d" %prediction)
					keyboard.release(translate[pressedKey])

				elif prediction in otherkey[pressedKey]: #Release and press new

					gotother = True

					#Make command
					keyboard.release(translate[pressedKey])
					print("Release key: %d" %pressedKey)

					#keyboard.press(translate[prediction])
					#print("Press key: %d" %prediction)
					#pressedKey = prediction

				if gotfive and ((gotopposite == True) or (gotother == True)):
					gotother = False
					gotopposite = False
					gotfive = False
					keypress = False
					print("Keypress = False")

				
			previousPrediction = prediction
		tme.sleep(0.05)

def mujaffaController():
	translate = {0:'x', 2:'down', 4:'left', 6:'right',8:'up'}
	opposite = {2:8, 4:6, 6:4, 8:2}
	keypress = False
	release = False
	while True:

		#Process commands
		lastprediction = glb.predictions[-1]
		#Statemachine
		if (lastprediction in [0,2,4,6,8]) and (keypress == False):
			pressedkey = lastprediction
			keypress=True

		if opposite[pressedkey] == lastprediction:
			release = True

		#Do actions
		if keypress:
			keyboard.press(translate[pressedkey])
			keypress = False

		if release:
			keyboard.release(translate[pressedkey])
			release = False
		tme.sleep(0.1)

def superMarioController():
	translate = {0:'space', 2:'down', 4:'left', 6:'right',8:'up'}
	opposite = {2:8, 4:6, 6:4, 8:2, 0:0}
	otherkey = {2:[4,6], 4:[2,8], 6:[2,8], 8:[4,6], 0:[2,4,6,8]}
	pressedkey = None
	keypress = False
	previousPrediction = 0
	while True:

		#Process commands
		if len(glb.predictions) > 1:
			with glb.predictionslock:
				prediction = glb.predictions[0]
				glb.predictions.pop(0)
			#Statemachine
			if (prediction in [2,4,6,8]) and (keypress == False): #Press
				pressedkey = prediction
				keyboard.press(translate[pressedkey])
				print("Pressed " + str(translate[pressedkey]))
				keypress = True

			if prediction == 0:
				if previousPrediction != 0:
					keyboard.press_and_release(translate[prediction])
					print("Press and release " + str(translate[prediction]))

			if (pressedkey != None) and (prediction != 0):

				if prediction == opposite[pressedkey]: #Release
					keypress = False
					keyboard.release(translate[pressedkey])
					print("Release " + str(translate[pressedkey]))

				elif prediction in otherkey[pressedkey]: #Release and press new
					keypress = True
					oldkey = copy.copy(pressedkey)
					pressedkey = prediction
					keyboard.release(translate[oldkey])
					keyboard.press(translate[pressedkey])
					print("Release "+ str(translate[oldkey])+"Press " + str(translate[pressedkey]))


			previousPrediction = prediction

		tme.sleep(0.01)
def tetrisController():
	translate = {0:'up', 2:'down', 4:'left', 6:'right',8:'down'}
	opposite = {2:8, 4:6, 6:4, 8:2, 0:0}
	otherkey = {2:[4,6], 4:[2,8], 6:[2,8], 8:[4,6], 0:[2,4,6,8]}
	pressedKey = None
	keypress = False
	previousPrediction = 0
	gotfive = False
	gotopposite = False
	gotother = False
	print("Starting tetrisController")
	while True:

		if len(glb.predictions) >= 1:
			with glb.predictionslock:
				prediction = copy.copy(glb.predictions[0])
				glb.predictions.pop(0)
			if (prediction != 5) and (prediction != previousPrediction):
				if (prediction in [4,6]) and (keypress == False): #Press
					keypress = True
					pressedKey = prediction
					keyboard.press_and_release(translate[prediction])
					print("Press and release " + str(translate[prediction]))
					#keyboard.press(translate[pressedKey])
					
					#print("Pressed " + str(translate[pressedKey]))
					
				
				elif prediction == 2:
					if previousPrediction != 2:
						#keyboard.press_and_release(translate[prediction])
						#print("Press and release " + str(translate[prediction]))
						pass
				elif prediction == 8: #Press and release
					if previousPrediction != 8:
						#keyboard.press_and_release(translate[prediction])
						#print("Press and release " + str(translate[prediction]))
						pass

				elif prediction == 0:
					if previousPrediction != 0: 
						keyboard.press_and_release(translate[prediction])
						print("Press and release " + str(translate[prediction]))
				
			if keypress and (prediction != previousPrediction):
				if prediction == 5:
					gotfive = True

				elif prediction == opposite[pressedKey]: #Release
					#keypress = False
					gotopposite = True
					print("Got opposite key: %d" %prediction)
					#keyboard.release(translate[pressedKey])
					#print("Release " + str(translate[pressedKey]))
				elif prediction in otherkey[pressedKey]: #Release and press new
					#keypress = True
					gotother = True
					#pass
					#print("Got other key: %d" %prediction)
					#keypress = False
					#oldkey = copy.copy(pressedKey)
					#pressedKey = prediction
					#keyboard.release(translate[oldkey])
					#keyboard.press(translate[pressedKey])
					#print("Release "+ str(translate[oldkey])+"Press " + str(translate[pressedKey]))
				if gotfive and ((gotopposite == True) or (gotother == True)):
					gotother = False
					gotopposite = False
					gotfive = False
					keypress = False
					print("Keypress = False")
				#elif ((gotopposite == True) and (gotother == True)):
					#gotother = False
					#gotopposite = False
					#gotfive = False
					#keypress = False
					#print("Keypress = False")
					#print("Duplicate prediction")
					'''
					elif prediction in otherkey[pressedkey]: #Release and press new
						keypress = True
						oldkey = copy.copy(pressedkey)
						pressedkey = prediction
						if pressedkey == 8:
							 keyboard.release(translate[oldkey])
							 keyboard.press_and_release(translate[pressedkey])
							 keypress = False
							 print("Release "+ str(translate[oldkey])+"Press and release " + str(translate[pressedkey]))
						else:
							keyboard.release(translate[oldkey])
							keyboard.press(translate[pressedkey])
							print("Release "+ str(translate[oldkey])+"Press " + str(translate[pressedkey]))
					'''
				
			previousPrediction = prediction
		tme.sleep(0.05)

def housekeeper():

	longsleep = False
	pop = False
	maxSizePredictions = 20
	while True:
		with glb.predictionslock:

			if len(glb.predictions) > maxSizePredictions:
				longsleep = False
				glb.predictions.pop(0)
			tme.sleep(0.01)
