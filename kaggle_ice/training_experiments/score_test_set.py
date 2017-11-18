import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from utilities import get_image_tensor
from define_models import create_model

score_data_location = 'data/test.hdf'

hyper_p = ('elu', 3, 2)
model = create_model(2, False, False)

model.load_weights('data/train_weightsnone_.hdf5')

model.compile(optimizer = Adam(), loss = 'binary_crossentropy',
	metrics = ['accuracy'])

submission_data = pd.read_hdf(score_data_location)
#submission_data.to_hdf('data/test.hdf', 'data', mode = 'w')
submission = pd.DataFrame()
submission['id'] = submission_data['id']
submission_data = get_image_tensor(submission_data, extra_channel = 'none')

results = model.predict(submission_data)
results = results.reshape(results.shape[0])

submission['is_iceberg'] = results
submission.to_csv('data/submission.csv', index = False)
#print(model.evaluate(x_test, test_target))
# Decent improvement so far, up to 71% test set