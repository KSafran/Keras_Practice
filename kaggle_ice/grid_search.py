import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

hyperparameters = [(a, k, p) for a in ('relu', 'elu') for k in (2, 3) for p in (2, 3)]

def get_model(hyper_p):
	model = Sequential()

	# Note - using elu rather than relu seems to improve thigs
	# this keeps neurons from 'dying', they still have a gradient when
	# the input is negative

	# Try copying this guy's layers to see if that is the issue
	# hypothesis 1, adding the third channel really matters
	# hypothesis 2, this guy's exact model architecture matters - doesn't help
	# hypothesis 3, relu is better than elu
	# hypothesis 4, batch sizes? - doesn't help
	# 5 just some bug I'm missing.

	model.add(Conv2D(64, kernel_size=(hyper_p[1],hyper_p[1]), activation = hyper_p[0], input_shape =((75,75,3))))
	model.add(MaxPooling2D(pool_size=(hyper_p[2],hyper_p[2]), strides=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(128, kernel_size=(hyper_p[1],hyper_p[1]), activation = hyper_p[0]))
	model.add(MaxPooling2D(pool_size=(hyper_p[2],hyper_p[2]), strides=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(128, kernel_size=(hyper_p[1],hyper_p[1]), activation = hyper_p[0]))
	model.add(MaxPooling2D(pool_size=(hyper_p[2],hyper_p[2]), strides=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, kernel_size=(hyper_p[1],hyper_p[1]), activation=hyper_p[0]))
	model.add(MaxPooling2D(pool_size=(hyper_p[2],hyper_p[2]), strides=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())

	# Let's add layers!!!
	model.add(Dense(512, activation = hyper_p[0]))
	model.add(Dropout(0.2))

	model.add(Dense(256, activation = hyper_p[0]))
	model.add(Dropout(0.2))

	model.add(Dense(1, activation = 'sigmoid'))

	# using this adam optimizer seems very popular
	adam_optimizer = mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	model.compile(optimizer = adam_optimizer, loss = 'binary_crossentropy',
	 metrics = ['accuracy'])
	return(model)

# format data for model
def shape_data(df):
	band_1 = np.array([np.array(x).astype(np.float32).reshape(75,75) for x in df['band_1']])
	band_2 = np.array([np.array(x).astype(np.float32).reshape(75,75) for x in df['band_2']])
	band_3 = (band_1 - band_2)
	all_bands_train = np.concatenate((band_1[:, :, :, np.newaxis], 
	band_1[:, :, :, np.newaxis], band_3[:, :, :, np.newaxis]),
	axis = 3)
	return(all_bands_train)

def normalize_image(array):
	return((array + 20) / 40)

# These callbacks should help with overfitting
def get_callbacks(filepath, patience = 2):
	early_stop = EarlyStopping('val_loss', patience = patience)
	model_save = ModelCheckpoint(filepath, save_best_only = True)
	return([early_stop, model_save])

iceberg = pd.read_json('data/train.json')

train_data, test_data = train_test_split(iceberg, train_size = .75, random_state = 100)

x_train = shape_data(train_data)
x_test = shape_data(test_data)

# We should normalize this data a bi
# min and max are around -45, 35, with mean -20
x_train = normalize_image(x_train)
x_test = normalize_image(x_test)

y_train = train_data.is_iceberg
y_test = test_data.is_iceberg

test_results = []
for hyp in hyperparameters:
	model = get_model(hyp)
	model.fit(x_train, y_train, epochs = 25, batch_size = 100,
		callbacks = get_callbacks('data/cv_weights.hdf5'),
		validation_data = (x_test, y_test))
	model.load_weights('data/cv_weights.hdf5')
	test_results.append(model.evaluate(x_test, y_test))

results = pd.DataFrame({'hyperparameters':hyperparameters,
	'scores':test_results})

results.to_csv('data/cv_results_2.csv', index=False)

# Decent improvement so far, up to 71% test set