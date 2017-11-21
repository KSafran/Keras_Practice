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