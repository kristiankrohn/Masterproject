import sys
sys.path.append('..')
import keyboard
import time as tme
key = None
oldKey = key
def print_pressed_keys(e):
	global key
	translateKey = {16:"q", 17:"w", 18:"e",
					30:"a", 31:"s", 32:"d",
					44:"z", 45:"x", 46:"c",
					57:" "}
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
keyboard.hook(print_pressed_keys)
#keyboard.wait()

while True:
	if key != oldKey:
		print(key)
		oldKey = key
	tme.sleep(0.01)
