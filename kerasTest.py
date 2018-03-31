import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

def baselineModel():
	#create the model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim = num_pixels, kernel_initializer = 'normal', activation = 'relu')) 
	model.add(Dense(num_classes, kernel_initializer = 'normal', activation = 'softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return model

if __name__ == '__main__':

	seed = 7
	np.random.seed(seed)

	(Xtrain, yTrain), (Xtest, yTest) = mnist.load_data()
	#reshape 28x28 pixels to a 784 flattened pixel array
	num_pixels = Xtrain.shape[1]*Xtrain.shape[2]
	Xtrain = Xtrain.reshape(Xtrain.shape[0], num_pixels).astype('float32')
	Xtest = Xtest.reshape(Xtest.shape[0], num_pixels).astype('float32')
	#Normalize the inputs, this is a greyscale between 0-255
	Xtrain = Xtrain / 255
	Xtest = Xtest / 255
	#One hot encoding of the classes
	yTrain = np_utils.to_categorical(yTrain)
	yTest = np_utils.to_categorical(yTest)
	num_classes = yTest.shape[1]

	model = baselineModel()
	model.fit(Xtrain, yTrain, validation_data = (Xtest, yTest), epochs = 10, batch_size = 200, verbose = 2)
	scores = model.evaluate(Xtest, yTest, verbose = 0)
	print('Baseline Error: %.2f%%' %(100 - scores[1]*100))
