from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten, concatenate, GlobalMaxPooling2D, Activation
from keras.models import Model

def create_model(nchannels = 2, base_size = 64, 
	 drop = 0.2, activation = 'elu', normalize_batches = False, 
	 angle = False):
	
	img = Input(shape = (75, 75, nchannels))
	
	x = Conv2D(base_size, (3,3))(img)
	# I've seen conflicting info about where to add batch norm
	# it looks like it was intended to be added before activation
	# to keep activation input near 0 where gradient wont vanish
	# also resnet does that, so I will too
	if normalize_batches: 
		x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	x = Dropout(drop)(x)
	

	x = Conv2D(2 * base_size, (3,3))(x)
	if normalize_batches:
		x = BatchNormalization()(x)
	x = Activation(activation)(x)	
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	x = Dropout(drop)(x)
	

	x = Conv2D(2 * base_size, (3,3))(x)
	if normalize_batches:
		x = BatchNormalization()(x)
	x = Activation(activation)(x)	
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	x = Dropout(drop)(x)
	if normalize_batches:
		x = BatchNormalization()(x)

	x = Conv2D(base_size, kernel_size=(3, 3))(x)
	if normalize_batches:
		x = BatchNormalization()(x)
	x = Activation(activation)(x)	
	x = GlobalMaxPooling2D()(x)
	x = Dropout(drop)(x)
	
	#x = Flatten()(x)

	# Let's add layers!!!
	x = Dense(8 * base_size)(x)
	if normalize_batches:
		x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = Dropout(drop)(x)

		
	x = Dense(4 * base_size)(x)
	if normalize_batches:
		x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = Dropout(drop)(x)

	if angle:
		angle_input = Input(shape = [1])

		both_ins = concatenate([x, angle_input], axis = -1)
		pred = Dense(1, activation = 'sigmoid')(both_ins)
		model = Model(inputs = [img, angle_input], outputs = pred)
	else:
		pred = Dense(1, activation = 'sigmoid')(x)
		model = Model(inputs = img, outputs = pred)
	
	return(model)