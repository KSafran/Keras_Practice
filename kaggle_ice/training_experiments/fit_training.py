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

pars = [(learn, batch) for learn in (0.001, 0.0005, 0.0002) for batch in (12, 24, 36)]
j = 0
# Run Experiments on
for p in pars:

	j += 1

	train_images = get_image_tensor(train_data, extra_channel = 'none')
	test_images = get_image_tensor(test_data, extra_channel = 'none')

	model = create_model(2, False, False)

	optimizer = mypotim=RMSprop(lr=p[0])
	model.compile(optimizer = optimizer, loss = 'binary_crossentropy',
	 metrics = ['accuracy'])

	hist = model.fit(train_images, 
		train_data.is_iceberg,
		epochs = 1,
		batch_size = p[1],
		callbacks = get_callbacks('data/train_weights_' + str(j) + '.hdf5', 6),
		validation_data = (test_images, test_data.is_iceberg))

	results = pd.DataFrame(hist.history)
	results['learn'] = p[0]
	results['batch_size'] = p[1]

	results.to_csv('data/hist_' + str(j) + '.csv')