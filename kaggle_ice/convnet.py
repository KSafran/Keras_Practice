import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

iceberg = pd.read_json('data/train.json')

train_data, test_data = train_test_split(iceberg, train_size = .7, random_state = 100)

model = Sequential()

model.add(Conv2D(50, (3,3), activation = 'relu', input_shape =((75,75,2))))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(25, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy',
 metrics = ['accuracy'])

# format data for model
def shape_data(df):
	band_1 = np.array([np.array(x).astype(np.float32).reshape(75,75) for x in df['band_1']])
	band_2 = np.array([np.array(x).astype(np.float32).reshape(75,75) for x in df['band_2']])
	both_bands_train = np.concatenate((band_1[:, :, :, np.newaxis], 
	band_1[:, :, :, np.newaxis]), axis = 3)
	return(both_bands_train)

x_train = shape_data(train_data)
x_test = shape_data(test_data)

train_target = train_data.is_iceberg
test_target = test_data.is_iceberg

model.fit(x_train, train_target, epochs = 5)

model.validate(x_test, test_target)
# benchmark accuracy of 44%, not great