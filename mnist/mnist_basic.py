import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import datasets, model_selection, preprocessing
import numpy as np
np.random.seed(100)

mnist = datasets.load_digits(10)

# need to one-hot encode the response
y = keras.utils.to_categorical(mnist.target, 10)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
	mnist.data, y, test_size=0.25, random_state=7)

# Initialize Model
mnist_model = Sequential()

# Add first dense layer. The first argument specifies the size
# of the layer (think # of nodes, or output size). Note that the 
# default activation function is just the identity function.
# You need to specify the input shape of the first layer. If we 
# were doing convolutions we would have to reshape this "flat" 
# image data, but for now we are just building a simple NN
mnist_model.add(Dense(32, input_shape=(64,)))
mnist_model.add(Activation('relu'))

# now add another layer
mnist_model.add(Dense(64))
mnist_model.add(Activation('relu'))

# now the output layer
mnist_model.add(Dense(10))
mnist_model.add(Activation('softmax'))

# Finally, compile the model and fit
mnist_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

mnist_model.fit(X_train, y_train, epochs = 20, batch_size = 128)
score = mnist_model.evaluate(X_test, y_test, batch_size=128)
print('''The out of sample cross entropy is %.3f and the out 
	of sample accuracy is %.3f''' % (score[0], score[1]))