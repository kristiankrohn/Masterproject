import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.wrappers.scikit_learn import KerasClassifier
import features
import predict
import classifier
import pyeeg
from sklearn import preprocessing
import itertools
import plot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.feature_selection import RFECV

#from numpy.fft import fft
#from numpy import zeros, floor
import math
import time
import dataset
from globalconst import  *
import globalvar



def createModel(numFeatures = 23, numClasses = 6):
	#create the model
	model = Sequential()
	model.add(Dense(numFeatures, input_dim = numFeatures, kernel_initializer = 'normal', activation = 'relu'))
	model.add(Dense(100, kernel_initializer  = 'normal', activation = 'relu'))
	model.add(Dense(10, kernel_initializer = 'normal', activation = 'relu'))
	model.add(Dense(numClasses, kernel_initializer = 'normal', activation = 'softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return model

def main():
	startLearning()

def startLearning():
	bestParams = []
	accuracyScore = []
    	f1Score = []
    	precision = []
    	classificationReport = []
	classifierstring = "learning260RBFsvm22Features"
	#Make sure that the result can be reproduced
	seed = 7
	np.random.seed(seed)

	X, y = dataset.loadDataset(filename="data.txt", filterCondition=True,
                                filterType="DcNotch", removePadding=True, shift=False, windowLength=250)
	X, y = dataset.sortDataset(X, y, length=10000, classes=[0,1,2,3,4,5,6,7,8,9], merge = True)
	#numClasses = 6
	channelIndex = 0

	featuremask = features.readFeatureMask()
	#Use number of features as input layer
	#numFeatures = len(featuremask)
	XL = features.extractFeaturesWithMask(
            X, channelIndex, featuremask=featuremask, printTime=False)

	XLtrain, XLtest, yTrain, yTest, XL, scaler = classifier.scaleAndSplit(XL, y[0])
	#One hot encoding of the classes
	#yTrain = np_utils.to_categorical(yTrain)
	#yTest = np_utils.to_categorical(yTest)
	#Define variable with number of classes
	clf = KerasClassifier(build_fn = createModel, epochs = 10, batch_size = 50, verbose = 0)
	#clf.fit(XLtrain, yTrain, validation_data = (XLtest, yTest), epochs = 10, batch_size = 200, verbose = 2)
	clf.fit(XLtrain, yTrain)
	#clf.fit(XLtrain, yTrain, validation_data = (XLtest, yTest), epochs = 10, batch_size = 50)

	#scores = model.evaluate(Xtest, yTest, verbose = 0)
	#print('Baseline Error: %.2f%%' %(100 - scores[1]*100))

	scores = cross_val_score(clf, XLtrain, yTrain, cv=50, scoring = 'accuracy')
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	print()
	print("Scores")
	print(scores)






if __name__ == '__main__':
	main()
