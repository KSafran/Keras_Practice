from keras.layers import Embedding, Dense, Activation, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import pandas as pd
from scipy.io import loadmat


# Load data
# this dataset comes form the coursera neural networks course
# taught by Geoffrey Hinton at U Toronto
# http://spark-public.s3.amazonaws.com/neuralnets/Programming%20Assignments/Assignment2/assignment2.zip
mat = loadmat('data/data.mat')
testData, trainData, validData, vocab = mat['data'][0, 0]

# We need to transpose these datasets
testData, trainData, validData = testData.transpose(), trainData.transpose(), validData.transpose()

model = Sequential()

# Embedding Layer
# input_dim needs to be the size of the vocabulary
# output_dim is just the dimension of the embedding
# input_length is the number of words in the sequence, 
# here we are just using a trigram to predict the 4th word
model.add(Embedding(input_dim = vocab.shape[1] + 1,
	output_dim = 50,
	input_length = 3))

model.add(Dense(200, activation = 'relu'))
model.add(Flatten())
model.add(Dense(251, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', 
	loss = 'categorical_crossentropy',
	metrics = ['accuracy'])

target = to_categorical(trainData[:,-1], num_classes = 251)

model.fit(trainData[:, [0,1,2]], target, epochs = 4, batch_size = 1000)

# Now check accuracy on test data
test_target = to_categorical(validData[:, -1], num_classes = 251)
score = model.validate(x_test, y_test)
print('''The out of sample cross entropy is %.3f and the out 
	of sample accuracy is %.3f''' % (score[0], score[1]))