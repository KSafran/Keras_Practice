from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten, Concatenate, GlobalMaxPooling2D
from keras.models import Model

# We want to include incidence angle so we need to use the 
# keras functional api rather than the sequential model

# Model structure inspired by github.com/cttsai1985
def create_model():
	image = Input(shape=[75, 75, 2])
	angle = Input(shape=[1])
	normalized_image = BatchNormalization()(image)

	# convs1
	x = Conv2D(filters= 8, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(normalized_image)
	x = Conv2D(filters= 8, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x)
	x = Conv2D(filters= 8, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((3, 3), (3,3))(x)
	x = Dropout(0.2)(x)

	#convs2
	x = Conv2D(filters= 16, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', 
		activation = 'elu')(x)
	x = Conv2D(filters= 16, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x)
	x = Conv2D(filters= 16, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((3, 3), (2,2))(x)
	x = Dropout(0.2)(x)

	# convs 3
	x = Conv2D(filters= 32, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x)
	x = Conv2D(filters= 32, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x)
	x = Conv2D(filters= 32, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((3, 3), (2,2))(x)
	x = Dropout(0.2)(x)

	# convs 4
	x = Conv2D(filters= 64, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x)
	x = Conv2D(filters= 64, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x)
	x = Conv2D(filters= 64, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x)
	x = Dropout(0.2)(x)
	x = GlobalMaxPooling2D()(x)
	x = BatchNormalization()(x)
	
	# Second Image Convs
	x2 = Conv2D(filters= 64, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(normalized_image)
	x2 = Conv2D(filters= 64, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x2)
	x2 = Conv2D(filters= 64, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x2)
	x2 = Conv2D(filters= 64, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x2)
	x2 = Conv2D(filters= 64, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x2)
	x2 = Conv2D(filters= 64, kernel_size = (3,3), 
		strides = (1,1), padding = 'same', activation = 'elu')(x2)
	x2 = Dropout(0.2)(x2)
	x2 = GlobalMaxPooling2D()(x2)
	x2= BatchNormalization()(x2)

	concat = Concatenate()([x, x2, angle])
	y = Dense(50, activation = 'elu')(concat)
	y = Dense(10, activation = 'elu')(y)
	predictions = Dense(1, activation = 'sigmoid')(y)

	model = Model(inputs = [image, angle], outputs = predictions)
	return(model)