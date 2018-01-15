def viewdataelement(index): #Utdatert
	global length
	file = open('Dataset\\data.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	i = index * 2
	if i > len(DataSet)-2:
		print("Value is too big")
	else:
		care = True
		feature1 = []
		feature1 = DataSet[i].split(',')
		featuretype1 = feature1[0]
		feature1.pop(0)
		if featuretype1 == 'u0':
			title = "Up"
		elif featuretype1 == 'd0':
			title = "Down"
		elif featuretype1 == 'l0':
			title = "Left"
		elif featuretype1 == 'r0':
			title = "Right"
		elif featuretype1 == 'c0':
			title = "Center"
			#care = False
		else:
			title = featuretype1
			care = False

		if care:
			plt.suptitle(title)
			print(title)
			print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-100,100])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("Fp1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-100,100])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("Fp2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			savestring = "Dataset_exports/figures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			#plt.savefig(savestring, bbox_inches='tight')
			plt.show()
			#plt.close()

def viewtempelement(index): #Utdatert
	global length
	file = open('Dataset\\temp.txt', 'r')
	AllData = file.read()
	DataSet = []
	DataSet = AllData.split(':')
	#print(DataSet)
	file.close()
	#DataSet.pop(-1)
	i = index * 2
	if i > len(DataSet)-2:
		print("Value is too big")
	else:
		care = True
		feature1 = []
		feature1 = DataSet[i].split(',')
		featuretype1 = feature1[0]
		feature1.pop(0)
		if featuretype1 == 'u0':
			title = "Up"
		elif featuretype1 == 'd0':
			title = "Down"
		elif featuretype1 == 'l0':
			title = "Left"
		elif featuretype1 == 'r0':
			title = "Right"
		elif featuretype1 == 'c0':
			title = "Center"
			#care = False
		else:
			title = featuretype1
			care = False

		if care:
			plt.suptitle(title)
			print(title)
			print(featuretype1)
			featureData1 = map(float, feature1)
			x = np.arange(0, length/250.0, 1.0/250.0)
			ax1 = plt.subplot(211)
			ax1.set_autoscaley_on(False)
			ax1.set_ylim([-100,100])
			plt.plot(x, featureData1, label=featuretype1)
			ax1.set_title("Fp1")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			feature2 = []
			feature2 = DataSet[i+1].split(',')
			featuretype2 = feature2[0]
			feature2.pop(0)
			print(featuretype2)
			featureData2 = map(float, feature2)
			ax2 = plt.subplot(212)
			ax2.set_autoscaley_on(False)
			ax2.set_ylim([-100,100])
			plt.plot(x, featureData2, label=featuretype2)
			ax2.set_title("Fp2")
			plt.ylabel('uV')
			plt.xlabel('Seconds')

			savestring = "figures/" + title +"/"+ title + str(i/2) + ".png"
			plt.subplots_adjust(hspace=0.45)
			#plt.savefig(savestring, bbox_inches='tight')
			plt.show()
			#plt.close()