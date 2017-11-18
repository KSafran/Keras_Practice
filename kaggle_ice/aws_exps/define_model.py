from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten, Concatenate, GlobalMaxPooling2D
from keras.models import Sequential

# We want to include incidence angle so we need to use the 
# keras functional api rather than the sequential model

# Model structure inspired by github.com/cttsai1985
def create_model():
	
	model = Sequential()

	model.add(Conv2D(64, (3,3), activation = 'elu', input_shape =((75,75,2))))
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
	return(model)