import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint

def normalize_image(array):
	return((array + 20) / 40)

def get_image_tensor(df, extra_channel = 'none', normalize = False):
	''' returns array of images
	df ~ the input dataframe
	extra_channel ~ one of ('none', 'diff', 'avg') for 3rd channel
	'''

	band_1 = np.array([np.array(x).astype(np.float32).reshape(75,75) for x in df['band_1']])
	band_2 = np.array([np.array(x).astype(np.float32).reshape(75,75) for x in df['band_2']])
	
	if extra_channel == 'avg':
		band_3 = (band_1 + band_2)/2
		all_bands =  np.concatenate((band_1[:, :, :, np.newaxis], 
		 band_2[:, :, :, np.newaxis], band_3[:, :, :, np.newaxis]),
		 axis = 3)
	elif extra_channel == 'diff':
		band_3 = (band_1 - band_2)
		all_bands =  np.concatenate((band_1[:, :, :, np.newaxis], 
		 band_2[:, :, :, np.newaxis], band_3[:, :, :, np.newaxis]),
		 axis = 3)
	else:
		all_bands = np.concatenate((band_1[:, :, :, np.newaxis], 
		band_2[:, :, :, np.newaxis]),
		axis = 3)
	
	if normalize:
		all_bands = normalize_image(all_bands)
	
	return(all_bands)

def get_callbacks(filepath, patience = 5):
	early_stop = EarlyStopping('val_loss', patience = patience)
	model_save = ModelCheckpoint(filepath, save_best_only = True)
	return([early_stop, model_save])

def get_hyperparameters(hyper_bounds):
	
	lr = np.power(10, np.random.uniform(low = hyper_bounds['log_lr_bounds'][0],
		high = hyper_bounds['log_lr_bounds'][1]))

	batch_size = np.random.random_integers(low = hyper_bounds['batch_bounds'][0],
		high = hyper_bounds['batch_bounds'][1])

	base_size = np.random.random_integers(low = hyper_bounds['base_size_bounds'][0],
		high = hyper_bounds['base_size_bounds'][1])

	activation = np.random.choice(hyper_bounds['activations'])

	bn = np.random.choice(hyper_bounds['bn_options'])

	dropout = np.random.uniform(low = hyper_bounds['dropout_bounds'][0],
		high = hyper_bounds['dropout_bounds'][1])

	return({'lr':lr,'batch_size':batch_size, 'base_size':base_size,
		'activation':activation, 'bn':bn, 'dropout':dropout})