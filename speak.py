from globalconst import  *
import globalvar as glb
import time as tme

'''
import os #Example for linux
from time import sleep
 
text = "text to speak"
 
cmd = 'espeak "{0}" 2>/dev/null'.format(text)
os.system(cmd)
sleep(1)
os.system(cmd)
'''

def Speak(sentence):
	with glb.speakLock:
	#print("Prediction lock")
		if not glb.speakQueue.full():            
			glb.speakQueue.put(sentence)
		else:
			popped = glb.speakQueue.get()
			glb.speakQueue.put(sentence)

def speakSystem():
	speakLib.Speak("Text to speech engine started")
	while True:
		try:
			with glb.speakLock:
				sentence = glb.speakQueue.get(block=False, timeout=1)
		except:
			sentence = None

		if sentence != None:	
			speakLib.Speak(sentence)
		else:
			tme.sleep(0.1)