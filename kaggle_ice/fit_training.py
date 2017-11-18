from keras.optimizers import Adam
import numpy as np
import pandas as pd
from model_copy import get_model
from utilities import get_image_tensor, get_callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split

model = get_model()
my_adam = Adam(lr=0.0005, beta_1 = 0.99, beta_2=0.999, epsilon = 1e-8, decay = 0.000)
model.compile(optimizer = my_adam, loss = 'binary_crossentropy', metrics = ['accuracy'])

iceberg = pd.read_json('data/train.json')
iceberg['inc_angle'] = iceberg['inc_angle'].replace('na', -1).astype(float)
train_data, test_data = train_test_split(iceberg, train_size = .75, random_state = 100)

train_images = get_image_tensor(train_data)
test_images = get_image_tensor(test_data)


hist = model.fit([train_images, train_data.inc_angle], 
	train_data.is_iceberg,
	epochs = 30,
	batch_size = 101,
	callbacks = get_callbacks('data/train_weights.hdf5', 3),
	validation_data = ([test_images, test_data.inc_angle], test_data.is_iceberg))

print(hist)