from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from attention_decoder import AttentionDecoder

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	# print('[randint(0, n_unique-1) for _ in range(length)]:')
	# print([randint(0, n_unique-1) for _ in range(length)])
	return [randint(0, n_unique-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	# print('[argmax(vector) for vector in encoded_seq]:')
	# print([argmax(vector) for vector in encoded_seq])
	return [argmax(vector) for vector in encoded_seq]

# prepare data for the LSTM
def get_pair(n_in, n_out, cardinality):
	# generate random sequence
	sequence_in = generate_sequence(n_in, cardinality)
	print("sequence_in")
	print(sequence_in)
	sequence_out = [0 for _ in range(n_in-n_out)]  + sequence_in[:n_out]
	print("sequence_out")
	print(sequence_out)
	# one hot encode
	X = one_hot_encode(sequence_in, cardinality)
	y = one_hot_encode(sequence_out, cardinality)
	# reshape as 3D
	X = X.reshape((1, X.shape[0], X.shape[1]))
	y = y.reshape((1, y.shape[0], y.shape[1]))
	"""
	print('X.shape:')
	print(X.shape)
	print('y.shape:')
	print(y.shape)
	print('X:')
	print(X)
	print('y:')
	print(y)
	"""
	return X,y

# define the encoder-decoder model
def baseline_model(n_timesteps_in, n_features):
	model = Sequential()
	model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
	model.add(RepeatVector(n_timesteps_in))
	model.add(LSTM(150, return_sequences=True))
	model.add(TimeDistributed(Dense(n_features, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

# define the encoder-decoder with attention model
def attention_model(n_timesteps_in, n_features):
	# n_timesteps_in = 5, n_features = 50
	model = Sequential()
	model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
	model.add(AttentionDecoder(150, n_features))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

# train and evaluate a model, return accuracy
def train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features):
	# train_evaluate_model(model, 5, 2, 50):
	# train LSTM
	for epoch in range(500):
		# generate new random sequence
		X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
		# fit model for one epoch on this sequence

		print('X.shape:')
		print(X.shape)
		print('y.shape:')
		print(y.shape)

		print('X:')
		print(X)
		print('y:')
		print(y)

		model.fit(X, y, epochs=1, verbose=1)
	# evaluate LSTM
	total, correct = 100, 0
	for _ in range(total):
		X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
		"""
		print('X:')
		print(X)
		print('y:')
		print(y)
		"""
		yhat = model.predict(X, verbose=1)
		if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
			correct += 1
	return float(correct)/float(total)*100.0

# configure problem
n_features = 9
n_timesteps_in = 5
n_timesteps_out = 1 # 2
n_repeats = 1
"""
# evaluate encoder-decoder model
print('Encoder-Decoder Model')
results = list()
for _ in range(n_repeats):
	model = baseline_model(n_timesteps_in, n_features)
	accuracy = train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features)
	results.append(accuracy)
	print(accuracy)
print('Mean Accuracy: %.2f%%' % (sum(results)/float(n_repeats)))
"""
# evaluate encoder-decoder with attention model
print('Encoder-Decoder With Attention Model')
results = list()
for _ in range(n_repeats):
	model = attention_model(n_timesteps_in, n_features)
	accuracy = train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features)
	results.append(accuracy)
	print(accuracy)
print('Mean Accuracy: %.2f%%' % (sum(results)/float(n_repeats)))
