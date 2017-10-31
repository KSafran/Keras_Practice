import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D
from sklearn import datasets, model_selection, preprocessing
import numpy as np
np.random.seed(100)

mnist = datasets.load_digits(10)

# need to one-hot encode the response
y = keras.utils.to_categorical(mnist.target, 10)

# Let's see how much improvement we can get by using 
# some convolutional layers. We need to reshape the 
# "flat" images into 8x8 images. Note the final parameter
# is the "channel". This is 1 for greyscale images, 3 for 
# rgb images.
training_shaped = mnist.data.reshape(mnist.data.shape[0], 8, 8, 1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
	training_shaped, y, test_size=0.25, random_state=7)

# Initialize Model
mnist_model = Sequential()

# Add first convolutional layer. The first argument specifies the size
# of the layer (think # of nodes, or output size). Note that the 
# default activation function is just the identity function.
# You need to specify the input shape of the first layer. Note the
# 1 on the end of the input shape is for the channel.
mnist_model.add(Conv2D(25, (3, 3), input_shape=(8,8,1), activation = 'relu'))
mnist_model.add(Conv2D(25, (3, 3), activation = 'relu'))
mnist_model.add(MaxPooling2D(pool_size=(2,2)))
mnist_model.add(Flatten())
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