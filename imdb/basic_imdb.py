import keras.datasets
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.preprocessing import sequence

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data('imdb.npz',
	num_words = 500,
	maxlen = 1000,
	seed=123,
	skip_top=5)

# the following makes all the various length movie reviews
# the same length by "padding" the beginning of the array
# with extra 0s (opposite of trimming)
x_train = sequence.pad_sequence(x_train)
x_test = sequence.pad_sequence(x_test)

imdb_model = Sequential()
imdb_model.add(Dense(200, input_shape(1,999), activation = 'relu')
imdb_model.add(Dense(1, activation = 'sigmoid'))
imdb_model.compile(opimier='rmsprop', 
	objective = 'binary-crossentropy', 
	metrics=['accuracy'])
imdb_model.fit(x_train, y_train)