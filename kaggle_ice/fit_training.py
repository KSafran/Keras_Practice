from keras.optimizers import Adam, RMSprop
import numpy as np
import pandas as pd
from define_model import create_model
from utilities import get_image_tensor, get_callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split

model = create_model(2, False, False)
optim = RMSprop(lr=0.0001)
model.compile(optimizer = optim, loss = 'binary_crossentropy', metrics = ['accuracy'])

iceberg = pd.read_json('data/train.json')
iceberg['inc_angle'] = iceberg['inc_angle'].replace('na', -1).astype(float)
train_data, test_data = train_test_split(iceberg, train_size = .75, random_state = 123)

train_images = get_image_tensor(train_data, 'none')
test_images = get_image_tensor(test_data, 'none')


hist = model.fit(train_images, 
	train_data.is_iceberg,
	epochs = 30,
	batch_size = 13,
	callbacks = get_callbacks('data/train_weights.hdf5', 5),
	validation_data = (test_images, test_data.is_iceberg))

print(hist)
pd.DataFrame(hist.history).to_csv('data/smaller_batch.csv')