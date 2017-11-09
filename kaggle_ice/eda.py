import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_json('data/train.json')

# Let's take a look at the picutres in different bands
# This one is a ship
# Band 1
ship_band_1 = np.ndarray(shape=[75,75], buffer=np.array(train_data.iloc[0,0]))
plt.imshow(ship_band_1)
plt.title('Ship - Band 1')
plt.show()

# Band 2
ship_band_2 = np.ndarray(shape=[75,75], buffer=np.array(train_data.iloc[0,1]))
plt.imshow(ship_band_2)
plt.title('Ship - Band 2')
plt.show()

# Now an iceberg
iceberg_band_1 = np.ndarray(shape=[75,75], buffer=np.array(train_data.iloc[2,0]))
plt.imshow(iceberg_band_1)
plt.title('Iceberg - Band 1')
plt.show()

# Band 2
iceberg_band_2 = np.ndarray(shape=[75,75], buffer=np.array(train_data.iloc[2,1]))
plt.imshow(iceberg_band_2)
plt.title('Band 2')
plt.show()