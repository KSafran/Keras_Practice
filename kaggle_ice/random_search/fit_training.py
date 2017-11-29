from keras.optimizers import Adam, RMSprop, Nadam, SGD
import numpy as np
import pandas as pd
from define_models import create_model
from utilities import get_image_tensor, get_callbacks, get_hyperparameters
from sklearn.model_selection import StratifiedKFold, train_test_split

hyperparameter_bounds = {'n_experiments' : 30,
 'log_lr_bounds' : [-3, -5],
 'batch_bounds' : [12, 64],
 'base_size_bounds' : [12, 140],
 'activations' : ('elu', 'relu', 'tanh'),
 'bn_options' : (True, False),
 'dropout_bounds' : [0.05, 0.5]
 }

# Set up data
iceberg = pd.read_json('data/train.json')
train_data, test_data = train_test_split(iceberg, train_size = .8, random_state = 123)

# Run Experiments on
for i in range(hyperparameter_bounds['n_experiments']):

	hyp = get_hyperparameters(hyperparameter_bounds)
	print(hyp)

	train_images = get_image_tensor(train_data, extra_channel = 'none', normalize = False)
	test_images = get_image_tensor(test_data, extra_channel = 'none', normalize = False)

	model = create_model(nchannels = 2, 
		base_size = hyp['base_size'],
		drop = hyp['dropout'],
		activation = hyp['activation'],
		normalize_batches = hyp['bn'],
		angle = False)

	model.compile(optimizer = Adam(lr = hyp['lr']), loss = 'binary_crossentropy',
	 metrics = ['accuracy'])

	hist = model.fit(train_images, 
		train_data.is_iceberg,
		epochs = 80,
		batch_size = hyp['batch_size'],
		callbacks = get_callbacks('data/train_weights_' + str(i) + '.hdf5', 6),
		validation_data = (test_images, test_data.is_iceberg))

	results = pd.DataFrame(hist.history)
	results['lr'] = hyp['lr']
	results['batch_size'] = hyp['batch_size']
	results['base_size'] = hyp['base_size']
	results['activation'] = hyp['activation']
	results['bn'] = hyp['bn']
	results['dropout'] = hyp['dropout']
	results['exp_num'] = i

	if i == 0:
		output = results
	else:
		output = output.append(results)

	
output.to_csv('data/exp_results.csv')