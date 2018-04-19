from globalconst import  *
import globalvar as glb
import keyboard
import time as tme
from datetime import datetime
from datetime import timedelta
import copy
import os
import sys; sys.path.append('.') # help python find ps_drone.py relative to scripts folder
sys.path.append('../ps_drone')
key = None
def print_pressed_keys(e):
	global key
	translateKey = {16:"q", 17:"w", 18:"e",
					30:"a", 31:"s", 32:"d",
					44:"z", 45:"x", 46:"c",
					57:" "} #Add more scancodes is needed
	line = [int(code) for code in keyboard._pressed_events]

	#print(line)
	
	if len(line)>0:
		try:
			key = translateKey[line[0]]
		except:
			key = "other"
	else:
		key = ""
	#print key

def droneController(debug=False):
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

	import ps_drone                                              # Import PS-Drone


	drone = ps_drone.Drone()                                     # Start using drone
	drone.startup()                                              # Connects to drone and starts subprocesses

	drone.reset()                                                # Sets drone's status to good (LEDs turn green when red)
	while (drone.getBattery()[0] == -1):      tme.sleep(0.1)    # Waits until drone has done its reset
	print "Battery: "+str(drone.getBattery()[0])+"%  "+str(drone.getBattery()[1])	# Gives a battery-status
	drone.useDemoMode(True)                                      # Just give me 15 basic dataset per second (is default anyway)

	##### Mainprogram begin #####
	drone.setConfigAllID()                                       # Go to multiconfiguration-mode
	'''
	drone.sdVideo()
	#drone.hdVideo()                                              # Choose lower resolution (hdVideo() for...well, guess it)
	drone.frontCam()                                             # Choose front view
	CDC = drone.ConfigDataCount
	while CDC == drone.ConfigDataCount:       tme.sleep(0.0001) # Wait until it is done (after resync is done)
	drone.startVideo()                                           # Start video-function
	drone.showVideo()                                            # Display the video
	'''
	drone.trim()                                       # Recalibrate sensors
	drone.getSelfRotation(5)                           # Get auto-alteration of gyroscope-sensor
	print "Auto-alt.:"+str(drone.selfRotation)+"dec/s" # Showing value for auto-alteration

	##### Controller variables ######
	opposite = {2:8, 4:6, 6:4, 8:2, 0:0}
	otherkey = {2:[4,6], 4:[2,8], 6:[2,8], 8:[4,6], 0:[2,4,6,8]}
	pressedKey = None
	keypress = False
	previousPrediction = 0
	prevPreviousPrediction = 0
	gotfive = 0
	gotopposite = 0
	gotother = 0
	#gotfive = 0
	blinks = 0
	lastTime = datetime.now()

	##### Flush prediction buffer before start
	with glb.predictionslock:
		del glb.predictions[:]
	##### And action !
	print "Use <x> to toggle front- and groundcamera, <space> to lift off, <z> to controll with brain, any other key to stop"
	IMC =    drone.VideoImageCount                               # Number of encoded videoframes
	stop =   False
	ground = False	
	drone.setSpeed(0.1)
	keyboard.hook(print_pressed_keys)
	while not stop:
		
		#while drone.VideoImageCount == IMC: tme.sleep(0.01)     # Wait until the next video-frame
		#IMC = drone.VideoImageCount
		
		#key = drone.getKey()        #this sux                             # Gets a pressed key

		if (key == " ") or (blinks >= 3):
			if drone.NavData["demo"][0][2] and not drone.NavData["demo"][0][3]:
				if debug:
					print("Takeoff")
				else:	
					drone.takeoff()

			else:
				if debug:
					print("land")
				else:	
					drone.land()
			blinks = 0
		
		#elif key=="x":
			#if ground:
				#ground = False
			#else:
				#ground = True
			#drone.groundVideo(ground)                            # Toggle between front- and groundcamera. Hint: options work for all videocommands
		
		elif key=="z":
			if brainz:
				brainz = False
				print("No longer flying with brainz")
				#drone.land()
			else:
				brainz = True
				print("Flying by brain, GO!")
				blinks = 0

			#drone.hover()
		

		if key == "":	drone.hover()
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
		elif key == "other":	stop = True

		#Brain controller


		if brainz:
			now = datetime.now()

			if (now - lastTime) > timedelta(seconds=3):
				if blinks > 0:
					blinks = blinks - 1
				else:
					blinks = 0
				lastTime = now

			if len(glb.predictions) >= 1:
				with glb.predictionslock:
					prediction = copy.copy(glb.predictions[0])
					glb.predictions.pop(0)

				#State transition from P0 to P1
				if prediction == 0:
					if previousPrediction == 0: 
						if prevPreviousPrediction != 0 and not keypress:
							blinks += 1
							print("Blinks: %d" %blinks)

				#if prediction != 5 and prediction != previousPrediction: 
				if not keypress and prediction == previousPrediction:	
					#State transition from S0 to S1
					#if prediction in [2,4,6,8] and keypress == False: 
					if prediction in [4,6,8] and keypress == False: 
						keypress = True
						pressedKey = prediction

						if prediction == 8:
							drone.moveForward()
							print("Move forwards")
						elif prediction == 2:
							drone.moveBackward()
							print("Move backwards")
						elif prediction == 4:
							drone.turnAngle(-30,1)
							print("Turn left")
						elif prediction == 6:
							print("Turn right")
							drone.turnAngle(30,1)
							
				if keypress:
				    #State transition from S1 to S2
					if prediction == opposite[pressedKey]:
					    if gotopposite == 0:
					        print("Hover")
					        drone.hover()
						gotopposite += 1
						
					elif prediction in otherkey[pressedKey]: 
						if otherkey == 0:
						    print("Hover")
						    drone.hover()
						gotother += 1 
											
					#State transistion from S2 to S0
					if prediction == 5:
						if (gotopposite == 1 or gotother == 1):
							gotother = 0
							gotopposite = 0
							gotfive = 0
							keypress = False
							print("Ready for new prediction, other/opposite exit")
						if pressedKey != 8:
							if ((previousPrediction == 5) and 
								(prevPreviousPrediction == 5)):
								print("Hover")
								drone.hover()
								gotother = 0
								gotopposite = 0
								gotfive = 0
								keypress = False
								print("Ready for new prediction, five exit")


				prevPreviousPrediction = previousPrediction
				previousPrediction = prediction
		#tme.sleep(0.01)
	print("Exiting drone controller")
	drone.land()

def onlineVerificationController():
	##### Controller variables ######
	#translate = {0:'up', 2:'down', 4:'left', 6:'right',8:'down'}
	
	opposite = {2:8, 4:6, 6:4, 8:2, 0:0}
	otherkey = {2:[4,6], 4:[2,8], 6:[2,8], 8:[4,6], 0:[2,4,6,8]}
	pressedKey = None
	keypress = False
	previousPrediction = 0
	prevPreviousPrediction = 0
	gotfive = 0
	gotopposite = 0
	gotother = 0
	#gotfive = 0
	blinks = 0
	lastTime = datetime.now()
	takeoff = False
	with glb.predictionslock:
		del glb.predictions[:]
	while True:
	
		if blinks >= 3:
			if not takeoff:
				print("Takeoff")
				takeoff = True
				#drone.takeoff()
			else:
				print("Land")
				takeoff = False
				#drone.land()
			blinks = 0
		
		now = datetime.now()

		if (now - lastTime) > timedelta(seconds=3):
			if blinks > 0:
				blinks = blinks - 1
			else:
				blinks = 0
			lastTime = now

		if len(glb.predictions) >= 1:
			with glb.predictionslock:
				prediction = copy.copy(glb.predictions[0])
				glb.predictions.pop(0)

			#State transition from P0 to P1
			if prediction == 0:
				if previousPrediction == 0: 
					if prevPreviousPrediction != 0 and not keypress:
						blinks += 1
						print("Blinks: %d" %blinks)

			#if prediction != 5 and prediction != previousPrediction: 
			if not keypress and prediction == previousPrediction:	
				#State transition from S0 to S1
				#if prediction in [2,4,6,8] and keypress == False: 
				if prediction in [4,6,8] and keypress == False: 
					keypress = True
					pressedKey = prediction

					if prediction == 8:
						#drone.moveForward()
						print("Move forwards")
					elif prediction == 2:
						#drone.moveBackward()
						print("Move backwards")
					elif prediction == 4:
						#drone.turnAngle(-30,1)
						print("Turn left")
					elif prediction == 6:
						print("Turn right")
						#drone.turnAngle(30,1)
						
			if keypress:
			    #State transition from S1 to S2
				if prediction == opposite[pressedKey]:
				    if gotopposite == 0:
				        print("Hover")
				        #drone.hover()
					gotopposite += 1
					
				elif prediction in otherkey[pressedKey]: 
					if otherkey == 0:
					    print("Hover")
					    #drone.hover()
					gotother += 1 
										
				#State transistion from S2 to S0
				if prediction == 5:
					if (gotopposite == 1 or gotother == 1):
						gotother = 0
						gotopposite = 0
						gotfive = 0
						keypress = False
						print("Ready for new prediction, other/opposite exit")
					if pressedKey != 8:
						if ((previousPrediction == 5) and 
							(prevPreviousPrediction == 5)):
							print("Hover")
							#drone.hover()
							gotother = 0
							gotopposite = 0
							gotfive = 0
							keypress = False
							print("Ready for new prediction, five exit")


			prevPreviousPrediction = previousPrediction
			previousPrediction = prediction

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
	print("Started predictions housekeeper")
	while True:
		with glb.predictionslock:

			if len(glb.predictions) > maxSizePredictions:
				longsleep = False
				glb.predictions.pop(0)
				#print("Popped predictions")
			tme.sleep(0.01)
