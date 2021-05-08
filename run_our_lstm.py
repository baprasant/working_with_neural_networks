from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
import os

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	# filepath = prefix + group + '/Inertial Signals/'
	current_dir = os.getcwd()
	if group is 'test':
		filepath = current_dir + '/neural_network_dataset/test/X_Test/'
	else:
		filepath = current_dir + '/neural_network_dataset/train/X_Train/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['x_axis_acc.txt', 'y_axis_acc.txt', 'z_axis_acc.txt']
	# body acceleration
	filenames += ['x_axis_gyro.txt', 'y_axis_gyro.txt', 'z_axis_gyro.txt']
	# body gyroscope
	filenames += ['x_axis_mag.txt', 'y_axis_mag.txt', 'z_axis_mag.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	# y = load_file(prefix + group + '/y_'+group+'.txt')
	if group is 'test':
		y = load_file(current_dir + '/neural_network_dataset/test/Y_Test.txt')
	else:
		y = load_file(current_dir + '/neural_network_dataset/train/Y_Train.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix="")
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix="")
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 1, 10, 10
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	print('trainx shape:')
	print(trainX.shape)
	print('n_timesteps, n_features, n_output')
	print(n_timesteps) # 128
	print(n_features) # 9
	print( n_outputs) # 6
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
	model.summary()
	y_pred = model.predict_classes(testX, verbose = 1)
	y_pred = y_pred[:0]
	'''
	print('ACCURACY')
	print(accuracy_score(testy, y_pred))
	print('PRECISION')
	print(precision_score(testy, y_pred))
	print('RECALL')
	print(recall_score(testy, y_pred))
	print('F1 SCORE')
	print(f1_score(testy, p_pred))
	'''
	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=1):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	print('trainX.shape')
	print(trainX.shape)
	print('trainy.shape')
	print(trainy.shape)
	print('testX.shape')
	print(testX.shape)
	print('testy.shape')
	print(testy.shape)
	print(testX[1])
	print('testX[1]')
	print(testy[1])
	print('testy[1]')
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)


# run the experiment
run_experiment()
