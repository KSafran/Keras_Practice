from keras.layers import Embedding, Dense, Activation, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import pandas as pd
from scipy.io import loadmat
import numpy as np

# Load data
# this dataset comes form the coursera neural networks course
# taught by Geoffrey Hinton at U Toronto
# http://spark-public.s3.amazonaws.com/neuralnets/Programming%20Assignments/Assignment2/assignment2.zip
mat = loadmat('data/data.mat')
testData, trainData, validData, vocab = mat['data'][0, 0]

# We need to transpose these datasets
testData, trainData, validData = testData.transpose(), trainData.transpose(), validData.transpose()

model = Sequential()

# Embedding Layer
# input_dim needs to be the size of the vocabulary
# output_dim is just the dimension of the embedding
# input_length is the number of words in the sequence, 
# here we are just using a trigram to predict the 4th word
model.add(Embedding(input_dim = vocab.shape[1] + 1,
	output_dim = 50,
	input_length = 3))

model.add(Dense(200, activation = 'relu'))
model.add(Flatten())
model.add(Dense(251, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', 
	loss = 'categorical_crossentropy',
	metrics = ['accuracy'])

#really only 250 classes, but the indices start at 1
target = to_categorical(trainData[:,-1], num_classes = 251)

model.fit(trainData[:, [0,1,2]], target, epochs = 4, batch_size = 1000)

# Now check accuracy on test data
test_target = to_categorical(validData[:, -1], num_classes = 251)
score = model.evaluate(validData[:, :-1], test_target)
print('''The out of sample cross entropy is %.3f and the out 
	of sample accuracy is %.3f''' % (score[0], score[1]))

# Can we see anything from our embedding layer?
from keras import backend as K

# This function returns the embedding output given a 3-gram
get_embedding = K.function([model.layers[0].input],
                                  [model.layers[0].output])

# we need to reshape the input here to fit with the embedding layer
layer_output = get_embedding([trainData[1,:-1].reshape([1,3])])

# how about a function that takes a word index and returns the embedding
def get_emb_index(ind):
	embedding_output = get_embedding([np.array([ind, ind, ind], ndmin = 2)])
	return(embedding_output[0][0,0,:])

# remember, word indexes start at 1
all_embeddings = [get_emb_index(i) for i in range(1, 251)]

def closest_word(index, embeddings):
	distances = [np.linalg.norm(embeddings[index] - embeddings[i]) for i in range(249)]
	ordering = np.argsort(distances)
	return(ordering[1:6])

def word_associations(index, embeddings, vocabulary):
	indices = closest_word(index, embeddings)
	words = [vocabulary[0][index].item()]
	words.append([vocabulary[0][j].item() for j in indices])
	return(words)

print('here are some examples of words that are associated with similar words')
print(word_associations(90, all_embeddings, vocab))
print(word_associations(10, all_embeddings, vocab))

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

word_tsne = TSNE(n_components = 2)
two_d_embedding = word_tsne.fit_transform(np.array(all_embeddings))
plt.scatter(two_d_embedding[:,0], two_d_embedding[:,1])
plt.show()