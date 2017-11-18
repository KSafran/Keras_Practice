from keras.optimizers import Adam, RMSprop
import numpy as np
import pandas as pd
from define_models import create_model
from utilities import get_image_tensor, get_callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split

# Set up data
iceberg = pd.read_json('data/train.json')
iceberg['inc_angle'] =  iceberg['inc_angle'].replace('na', -1)
train_data, test_data = train_test_split(iceberg, train_size = .75, random_state = 123)

pars = [(chan, norm, angle) for chan in ('none', 'diff', 'avg') for angle in (True, False) for norm in (True, False)]

# Run Experiments on
for p in pars:
	train_images = get_image_tensor(train_data, extra_channel = p[0])
	test_images = get_image_tensor(test_data, extra_channel = p[0])

	if p[0] == 'none':
		model = create_model(2, p[1], p[2])
	else:
		model = create_model(3, p[1], p[2])

	optimizer = mypotim=RMSprop(lr=0.001)
	model.compile(optimizer = optimizer, loss = 'binary_crossentropy',
	 metrics = ['accuracy'])

	pars_suffix = p[0] + '_' + 'norm_' * p[1] + 'angle_' * p[2]
	print(pars_suffix)

	if p[2]:
		hist = model.fit([train_images, train_data.inc_angle], 
			train_data.is_iceberg,
			epochs = 50,
			batch_size = 102,
			callbacks = get_callbacks('data/train_weights' + pars_suffix + '.hdf5', 6),
			validation_data = ([test_images, test_data.inc_angle], test_data.is_iceberg))
	else:
		hist = model.fit(train_images, 
			train_data.is_iceberg,
			epochs = 50,
			batch_size = 102,
			callbacks = get_callbacks('data/train_weights' + pars_suffix + '.hdf5', 6),
			validation_data = (test_images, test_data.is_iceberg))

	pd.DataFrame(hist.history).to_csv('data/hist_rms_' +pars_suffix + '.csv')