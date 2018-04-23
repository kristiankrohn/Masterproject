from datetime import datetime
import time as tme
import psutil, os

def setpriority(priority=2):
	p = psutil.Process(os.getpid())
	print("Original priority:")
	print(p.nice())

	priorityclasses = [psutil.IDLE_PRIORITY_CLASS,
						psutil.BELOW_NORMAL_PRIORITY_CLASS,
						psutil.NORMAL_PRIORITY_CLASS,
						psutil.ABOVE_NORMAL_PRIORITY_CLASS,
						psutil.HIGH_PRIORITY_CLASS,
						psutil.REALTIME_PRIORITY_CLASS]

	p.nice(priorityclasses[priority])
	print("New priority:")
	print(p.nice())

def main():
	for j in range(5):
		print("Set priority to %d" %j)
		setpriority(priority=j)
		#lastTime = datetime.now()
		for i in range(100):
			start = datetime.now()
			tme.sleep(0.001)
			stop = datetime.now()
			print(stop-start)

			'''
			now = datetime.now()
			if lastTime != now:
				print(now)
				lastTime = now
			'''

if __name__ == '__main__':
    main()
