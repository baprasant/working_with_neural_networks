from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from matplotlib import pyplot
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal


class HAR(object):

    def __init__(self):
        self.trainX = None
        self.trainy = None
        self.testX = None
        self.testy = None
        self.model = None
        real_time_data = None
        self.epochs=10
        self.batch_size=32
        self.verbose=1


    def __del__(self):
        self.trainX = None
        self.trainy = None
        self.testX = None
        self.testy = None
        self.model = None
        self.epochs=None
        self.batch_size=None
        self.verbose=None
        real_time_data = None
        del real_time_data
        del self.trainX
        del self.trainy
        del self.testX
        del self.testy
        del self.model
        del self.epochs
        del self.batch_size
        del self.verbose


    def run_experiment(self, repeats=1):
        # load data
        self.load_dataset()
        scores = list()
        for r in range(repeats):
            score = self.evaluate_model()
            score = score * 100.0
            print('>#%d: %.3f' % (r+1, score))
            scores.append(score)
        # summarize results
        self.summarize_results(scores)


    def summarize_results(self, scores):
        print(scores)
        m, s = mean(scores), std(scores)
        print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


    def evaluate_model(self):
        # define model
        n_timesteps, n_features, n_outputs = self.trainX.shape[1], self.trainX.shape[2], self.trainy.shape[1]
        # reshape data into time steps of sub-sequences
        n_steps, n_length = 4, 32
        trainX = self.trainX.reshape((self.trainX.shape[0], n_steps, n_length, n_features))
        testX = self.testX.reshape((self.testX.shape[0], n_steps, n_length, n_features))
        # define model
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, strides = 1, activation='relu'), input_shape=(None,n_length,n_features)))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(Bidirectional(LSTM(50)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        # model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model.fit(trainX, self.trainy, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # evaluate model
        print('Evaluating...')
        _, accuracy = model.evaluate(testX, self.testy, batch_size=self.batch_size, verbose=self.verbose)
        print('Predicting...')
        y_pred = model.predict_classes(testX[0:10], verbose = self.verbose)
        self.model = model
        return accuracy


    # load a single file as a numpy array
    def load_file(self, filepath):
        dataframe = read_csv(filepath, header=None, delim_whitespace=True)
        return dataframe.values


    # load a list of files and return as a 3d numpy array
    def load_group(self, filenames, prefix=''):
        loaded = list()
        for name in filenames:
            data = self.load_file(prefix + name)
            loaded.append(data)
        # stack group so that features are the 3rd dimension
        loaded = dstack(loaded)
        return loaded


    # load a dataset group, such as train or test
    def load_dataset_group(self, group, prefix=''):
        filepath = prefix + group + '/Inertial Signals/'
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
        # body acceleration
        filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
        # body gyroscope
        filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
        # load input data
        X = self.load_group(filenames, filepath)
        # load class output
        y = self.load_file(prefix + group + '/y_'+group+'.txt')
        return X, y


    # load the dataset, returns train and test X and y elements
    def load_dataset(self, prefix=''):
        # load all train
        trainX, trainy = self.load_dataset_group('train', prefix + 'HARDataset/')
        print(trainX.shape, trainy.shape)
        # load all test
        testX, testy = self.load_dataset_group('test', prefix + 'HARDataset/')
        print(testX.shape, testy.shape)
        # zero-offset class values
        trainy = trainy - 1
        testy = testy - 1
        # one hot encode y
        trainy = to_categorical(trainy)
        testy = to_categorical(testy)
        self.trainX = trainX
        self.trainy = trainy
        self.testX = testX
        self.testy = testy


    def get_real_time_data(self):
        prefix = 'RealTimeData/'
        filepath = prefix
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['total_acc_x_.txt', 'total_acc_y_.txt', 'total_acc_z_.txt']
        # body acceleration
        filenames += ['body_acc_x_.txt', 'body_acc_y_.txt', 'body_acc_z_.txt']
        # body gyroscope
        filenames += ['body_gyro_x_.txt', 'body_gyro_y_.txt', 'body_gyro_z_.txt']
        # load input data
        loaded = list()
        for name in filenames:
            data = self.load_file(prefix + name)
            loaded.append(data)
        # stack group so that features are the 3rd dimension
        loaded = dstack(loaded)
        n_timesteps, n_features = loaded.shape[1], loaded.shape[2]
        # reshape data into time steps of sub-sequences
        n_steps, n_length = 4, 32
        self.real_time_data = loaded.reshape((loaded.shape[0], n_steps, n_length, n_features))


    def real_time_prediction_class(self):
        print('Predicting based on real time data...')
        y_pred = self.model.predict_classes(self.real_time_data, verbose = self.verbose)
        return y_pred


    def real_time_prediction(self):
        print('Predicting based on real time data...')
        y_pred = self.model.predict(self.real_time_data, verbose = self.verbose)
        return y_pred


def main():
    har = HAR()
    har.run_experiment()
    har.model.summary()
    har.get_real_time_data()
    pred = har.real_time_prediction_class()
    print('pred class:')
    print(pred+1)
    pred = har.real_time_prediction()
    print('pred:')
    print(pred[0])
    print('in float')
    for each_ in pred[0]:
        print(str(float(each_)))


if __name__ == '__main__':
    main()
