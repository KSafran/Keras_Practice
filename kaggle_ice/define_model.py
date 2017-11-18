from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten, concatenate, GlobalMaxPooling2D
from keras.models import Model

def create_model(nchannels = 2, normalize_batches = False, angle = False):
	
	img = Input(shape = (75, 75, nchannels))
	
	x = Conv2D(64, (3,3), activation = 'elu')(img)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	x = Dropout(0.2)(x)
	if normalize_batches:
		x = BatchNormalization()(x)

	x = Conv2D(128, (3,3), activation = 'elu')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	x = Dropout(0.2)(x)
	if normalize_batches:
		x = BatchNormalization()(x)

	x = Conv2D(64, kernel_size=(3, 3), activation='elu')(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	x = Dropout(0.2)(x)
	if normalize_batches:
		x = BatchNormalization()(x)

	x = Flatten()(x)

	# Let's add layers!!!
	x = Dense(512, activation = 'elu')(x)
	x = Dropout(0.2)(x)
	if normalize_batches:
		x = BatchNormalization()(x)
		
	x = Dense(256, activation = 'elu')(x)
	x = Dropout(0.2)(x)

	if angle:
		angle_input = Input(shape = [1])

		both_ins = concatenate([x, angle_input], axis = -1)
		pred = Dense(1, activation = 'sigmoid')(both_ins)
		model = Model(inputs = [img, angle_input], outputs = pred)
	else:
		pred = Dense(1, activation = 'sigmoid')(x)
		model = Model(inputs = img, outputs = pred)
	
	return(model)