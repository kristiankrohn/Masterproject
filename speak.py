from globalconst import  *
import globalvar as glb
import time as tme
import os #Example for linux

 
#from gtts import gTTS

#tts = gTTS(text='Good morning', lang='en')


def Speak(sentence):
	with glb.speakLock:
	#print("Prediction lock")
		if not glb.speakQueue.full():            
			glb.speakQueue.put(sentence)
		else:
			popped = glb.speakQueue.get()
			glb.speakQueue.put(sentence)

def speakSystem():
	if os.name == 'nt':	
		speakLib.Speak("Text to speech engine started")
	elif os.name == 'posix':
		os.system('espeak "{0}" 2>/dev/null'.format("Text to speech engine started"))	
	while True:
		try:
			with glb.speakLock:
				sentence = glb.speakQueue.get(block=False, timeout=1)
		except:
			sentence = None

		if sentence != None:
			if os.name == 'nt':	
				speakLib.Speak(sentence)
			elif os.name == 'posix':
				
				os.system('espeak "{0}" 2>/dev/null'.format(sentence))
				#os.system('espeak -ven-en+f1 -s170'.format(sentence))
		else:	
			tme.sleep(0.05)