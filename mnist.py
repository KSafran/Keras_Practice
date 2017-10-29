from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import datasets

mnist = datasets.load_digits([10])