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
import time
import xlwt
from xlwt import Workbook

from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import RepeatVector
from attention_decoder import AttentionDecoder
from keras.callbacks import Callback

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

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
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    # print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    # print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    # print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    row_number = 0
    wb = Workbook()
    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('LSTM And CNN Observations')
    sheet1.write(row_number, 0, 'EPOCH')
    sheet1.write(row_number, 1, 'HIDDEN LAYERS')
    sheet1.write(row_number, 2, 'ACC AT FIRST EPOCH')
    sheet1.write(row_number, 3, 'ACC AT LAST EPOCH')
    sheet1.write(row_number, 4, 'CV ACC')
    sheet1.write(row_number, 5, 'TDV ACC')
    sheet1.write(row_number, 6, 'MODEL SIZE')
    sheet1.write(row_number, 7, 'TRAINING TIME')
    sheet1.write(row_number, 8, 'FILTERS')
    sheet1.write(row_number, 9, 'STRIDES')
    sheet1.write(row_number, 10, 'BATCH SIZE')
    n_steps, n_length = 4, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    ep_values = [7]
    for ep in ep_values: # for ep in range(7,11):
        # print('With ep:')
        # print(ep)
        filters_ = [32]
        strides_ = [1,5]
        for stride_ in strides_:
            for filter_ in filters_:
                verbose, epochs= 2, ep
                batch_size_ = 32
                # print('n_timesteps, n_features, n_output')
                # print(n_timesteps) # 128
                # print(n_features) # 9
                # print( n_outputs) # 6
                for hidden_layers in range(20,60,20):
                    scores_cv = list()
                    scores_tdv = list()
                    # print('With hidden_layers:')
                    # print(hidden_layers)
                    print('-------------------------------------------------------------------')
                    print('Number of Epochs:'+str(ep))
                    print('Number of Hidden Layers: '+str(hidden_layers))
                    print('Filters:'+str(filter_))
                    print('Strides:'+str(stride_))
                    model = Sequential()
                    model.add(TimeDistributed(Conv1D(filters=filter_, kernel_size=3, strides = stride_, activation='relu'), input_shape=(None,n_length,n_features)))
                    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
                    model.add(TimeDistributed(Flatten()))
                    model.add(LSTM(hidden_layers))
                    model.add(Dense(100, activation='relu'))
                    model.add(Dense(n_outputs, activation='softmax'))
                    #  model.summary()
                    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    # fit network
                    time_callback = TimeHistory()
                    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size_, verbose=verbose, callbacks=[time_callback])
                    # evaluate model
                    print('Cross Validation...')
                    _, accuracy_cv = model.evaluate(trainX, trainy, batch_size=batch_size_, verbose=2)
                    print('Test Data Validation...')
                    _, accuracy_tdv = model.evaluate(testX, testy, batch_size=batch_size_, verbose=2)
                    # print('Predicting...')
                    # y_pred = model.predict_classes(testX[0:10], verbose = 1)
                    # model.save("current_model.h5")
                    # print("Saved model to disk")
                    # f_size = 'NA'
                    # print("File size in bytes of model: ",file_size("current_model.h5"))
                    # os.system('rm current_model.h5')
                    """
                    print("x_pred[0]:")
                    print(testX[0])
                    print("y_pred_class:")
                    print(y_pred[0])
                    """
                    score_cv = accuracy_cv * 100.0
                    score_tdv = accuracy_tdv * 100.0
                    # score = score * 100.0
                    r = 0
                    # print('>#%d: %.3f' % (r+1, score_cv))
                    # print('>#%d: %.3f' % (r+1, score_tdv))
                    scores_cv.append(score_cv)
                    scores_tdv.append(score_tdv)
                    # summarize results
                    acc_cv = summarize_results_cv(scores_cv)
                    acc_tdv = summarize_results_tdv(scores_tdv)
                    training_time = sum(time_callback.times)
                    print('Training Time:'+str(training_time))
                    acc_at_last_epoch = history.history['accuracy'][-1]*100
                    print('Accuracy at Last Epoch:'+str(acc_at_last_epoch))
                    acc_at_first_epoch = history.history['accuracy'][0]*100
                    print('Accuracy at First Epoch:'+str(acc_at_first_epoch))
                    row_number = row_number + 1
                    sheet1.write(row_number, 0, ep)
                    sheet1.write(row_number, 1, hidden_layers)
                    sheet1.write(row_number, 2, acc_at_first_epoch)
                    sheet1.write(row_number, 3, acc_at_last_epoch)
                    sheet1.write(row_number, 4, acc_cv)
                    sheet1.write(row_number, 5, acc_tdv)
                    sheet1.write(row_number, 6, 'NA')
                    sheet1.write(row_number, 7, training_time)
                    sheet1.write(row_number, 8, filter_)
                    sheet1.write(row_number, 9, stride_)
                    sheet1.write(row_number, 10, batch_size_)
                    print('-------------------------------------------------------------------')
    wb.save('LSTM_CNN_REPORT_BS_32_RPI4.xls')

# summarize scores
def summarize_results_cv(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy for CV: %.3f%% (+/-%.3f)' % (m, s))
    return m

# summarize scores
def summarize_results_tdv(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy for TDV: %.3f%% (+/-%.3f)' % (m, s))
    return m

# run an experiment
def run_experiment(repeats=1):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    for r in range(repeats):
        evaluate_model(trainX, trainy, testX, testy)

def file_size(fname):
        statinfo = os.stat(fname)
        return statinfo.st_size

# run the experiment
run_experiment()
