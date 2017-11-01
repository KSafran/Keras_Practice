import keras.datasets
from keras.layers import Dense, Activation
from keras.models import Sequential

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data('imdb.npz',
	num_words = 500,
	maxlen = 1000,
	seed=123,
	skip_top=5)
