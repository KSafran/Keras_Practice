import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, concatenate, Flatten, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16

iceberg = pd.read_json('data/train.json')
iceberg.inc_angle = iceberg.inc_angle.replace('na', 0)

train_data, test_data = train_test_split(iceberg, train_size = .75, random_state = 123)



angle_input = Input(shape  = [1])

vgg = VGG16(weights = 'imagenet', include_top = False,
	input_shape = [75, 75, 3], classes = 1)

x = vgg.get_layer('block5_pool').output

x = GlobalMaxPooling2D()(x)

x = concatenate([x, angle_input])
x = Dense(512, activation = 'elu')(x)
x = Dropout(0.2)(x)

x = Dense(256, activation = 'elu')(x)
x = Dropout(0.2)(x)

output = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs = [vgg.input, angle_input], outputs = output)
# using this adam optimizer seems very popular
adam_optimizer = mypotim=Adam(lr=0.001)

model.compile(optimizer = adam_optimizer, loss = 'binary_crossentropy',
 metrics = ['accuracy'])

# format data for model
def shape_data(df):
	band_1 = np.array([np.array(x).astype(np.float32).reshape(75,75) for x in df['band_1']])
	band_2 = np.array([np.array(x).astype(np.float32).reshape(75,75) for x in df['band_2']])
	band_3 = (band_1 + band_2)
	all_bands_train = np.concatenate((band_1[:, :, :, np.newaxis], 
	band_2[:, :, :, np.newaxis], band_3[:, :, :, np.newaxis]),
	axis = 3)
	return(all_bands_train)

x_train = shape_data(train_data)
x_test = shape_data(test_data)

train_target = train_data.is_iceberg
test_target = test_data.is_iceberg

# We can improve generalization with a data generator
# this will create "new" images by transforming our 
# training set

# These callbacks should help with overfitting
def get_callbacks(filepath, patience = 5):
	early_stop = EarlyStopping('val_loss', patience = patience)
	model_save = ModelCheckpoint(filepath, save_best_only = True)
	return([early_stop, model_save])

model.fit([x_train, train_data.inc_angle], train_target, batch_size = 64, 
	epochs = 50, 
	callbacks = get_callbacks('data/model_weights.hdf5'),
	validation_data = ([x_test, test_data.inc_angle], test_target))

print(model.evaluate(x_test, test_target))
model.load_weights('data/model_weights.hdf5')
print(model.evaluate(x_test, test_target))
# Decent improvement so far, up to 71% test set