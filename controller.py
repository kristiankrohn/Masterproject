from globalconst import  *
import globalvar as glb
import keyboard
import time as tme
import copy

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
