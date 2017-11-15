import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

iceberg = pd.read_json('data/train.json')

train_data, test_data = train_test_split(iceberg, train_size = .75, random_state = 100)

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

model.add(Conv2D(64, (3,3), activation = 'elu', input_shape =((75,75,3))))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation = 'elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation = 'elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

# Let's add layers!!!
model.add(Dense(512, activation = 'elu'))
model.add(Dropout(0.2))

model.add(Dense(256, activation = 'elu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation = 'sigmoid'))

# using this adam optimizer seems very popular
adam_optimizer = mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer = adam_optimizer, loss = 'binary_crossentropy',
 metrics = ['accuracy'])

# format data for model
def shape_data(df):
	band_1 = np.array([np.array(x).astype(np.float32).reshape(75,75) for x in df['band_1']])
	band_2 = np.array([np.array(x).astype(np.float32).reshape(75,75) for x in df['band_2']])
	band_3 = band_1 - band_2
	all_bands_train = np.concatenate((band_1[:, :, :, np.newaxis], 
	band_1[:, :, :, np.newaxis], band_3[:, :, :, np.newaxis]),
	axis = 3)
	return(all_bands_train)

x_train = shape_data(train_data)
x_test = shape_data(test_data)

# We should normalize this data a bi
# min and max are around -45, 35, with mean -20

def normalize_image(array):
	return((array + 20) / 40)

x_train = normalize_image(x_train)
x_test = normalize_image(x_test)

train_target = train_data.is_iceberg
test_target = test_data.is_iceberg

# We can improve generalization with a data generator
# this will create "new" images by transforming our 
# training set
datagen = ImageDataGenerator(rotation_range=360,
	horizontal_flip=True,
	vertical_flip=True)

# These callbacks should help with overfitting
def get_callbacks(filepath, patience = 2):
	early_stop = EarlyStopping('val_loss', patience = patience)
	model_save = ModelCheckpoint(filepath, save_best_only = True)
	return([early_stop, model_save])

model.fit_generator(datagen.flow(x_train, train_target, batch_size = 100), 
	epochs = 15,	steps_per_epoch = 12,
	callbacks = get_callbacks('data/model_weights.hdf5'),
	validation_data = (x_test, test_target))

print(model.evaluate(x_test, test_target))
model.load_weights('data/model_weights.hdf5')
print(model.evaluate(x_test, test_target))
# Decent improvement so far, up to 71% test set