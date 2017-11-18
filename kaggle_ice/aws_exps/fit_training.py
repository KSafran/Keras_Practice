from keras.optimizers import Adam, RMSprop
import numpy as np
import pandas as pd
from define_model import create_model
from utilities import get_image_tensor, get_callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split

# Set up data
iceberg = pd.read_json('data/train.json')
train_data, test_data = train_test_split(iceberg, train_size = .75, random_state = 123)

train_images = get_image_tensor(train_data)
test_images = get_image_tensor(test_data)

# Run Experiments on adam lr
lrs = [0.002, 0.001, 0.0005]
for i in range(3):
	model = create_model()
	optimizer = Adam(lr=lrs[i])
	model.compile(optimizer = optimizer, loss = 'binary_crossentropy',
	 metrics = ['accuracy'])

	hist = model.fit(train_images, 
		train_data.is_iceberg,
		epochs = 1,
		batch_size = 101,
		callbacks = get_callbacks('data/train_weights_adam' + str(i) + '.hdf5', 4),
		validation_data = (test_images, test_data.is_iceberg))

	pd.DataFrame(hist.history).to_csv('data/hist_adam_' + str(i) + '.csv')

# Run Experiments on rmsprop lr
for i in range(3):
	model = create_model()
	optimizer = mypotim=RMSprop(lr=lrs[i])
	model.compile(optimizer = optimizer, loss = 'binary_crossentropy',
	 metrics = ['accuracy'])

	hist = model.fit(train_images, 
		train_data.is_iceberg,
		epochs = 1,
		batch_size = 101,
		callbacks = get_callbacks('data/train_weights_rms_' + str(i) + '.hdf5', 4),
		validation_data = (test_images, test_data.is_iceberg))

	pd.DataFrame(hist.history).to_csv('data/hist_rms_' + str(i) + '.csv')