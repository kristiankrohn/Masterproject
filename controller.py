from globalconst import  *
import globalvar as glb
import keyboard
import time as tme

def mujaffaController():
	translate = {0:'x', 2:'down', 4:'left', 6:'right',8:'up'}
	while True:
		
		#Process commands
		lastprediction = glb.predictions[-1]
		#Statemachine
		if lastprediction in [0,2,4,6,8]:
			keypress=True
		else:
			keypress=False
		#Do actions
		if keypress:
			keyboard.press_and_release(translate[lastprediction])
			#keyboard.press()
		tme.sleep(0.1)

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
