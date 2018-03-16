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
