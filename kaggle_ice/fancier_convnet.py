import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

iceberg = pd.read_json('data/train.json')

train_data, test_data = train_test_split(iceberg, train_size = .7, random_state = 100)

model = Sequential()

# Note - using elu rather than relu seems to improve thigs
# this keeps neurons from 'dying', they still have a gradient when
# the input is negative

# quite a bit of improvment from adding these layers with 
# more nodes
model.add(Conv2D(100, (4,4), activation = 'elu', input_shape =((75,75,2))))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(100, (3,3), activation = 'elu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(50, (3,3), activation = 'elu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())

# Let's add layers!!!
model.add(Dense(50, activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation = 'elu'))

model.add(Dense(1, activation = 'sigmoid'))

# using this adam optimizer seems very popular
adam_optimizer = mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer = adam_optimizer, loss = 'binary_crossentropy',
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

# We should normalize this data a bi
# min and max are around -45, 35, with mean -20

def normalize_image(array):
	return((array + 20) / 40)

x_train = normalize_image(x_train)
x_test = normalize_image(x_test)

train_target = train_data.is_iceberg
test_target = test_data.is_iceberg

model.fit(x_train, train_target, epochs = 5, batch_size = 100)

print(model.evaluate(x_test, test_target))
# Decent improvement so far, up to 71% test set