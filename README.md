# working_with_neural_networks

## Neural Network
A neural network is a type of machine learning which models itself after the human brain. This creates an artificial neural network that via an algorithm allows the computer to learn by incorporating new data.

## Machine Learning
A subset of artificial intelligence involved with the creation of algorithms which can modify itself without human intervention to produce desired output- by feeding itself through structured data.

## Deep Learning
A subset of machine learning where algorithms are created and function similar to those in machine learning, but there are numerous layers of these algorithms- each providing a different interpretation to the data it feeds on. Such a network of algorithms are called artificial neural networks, being named so as their functioning is an inspiration, or you may say; an attempt at imitating the function of the human neural networks present in the brain.

## Convolution in Convolutional Neural Networks
The convolutional neural network, or CNN for short, is a specialized type of neural network model designed for working with two-dimensional image data, although they can be used with one-dimensional and three-dimensional data.

Central to the convolutional neural network is the convolutional layer that gives the network its name. This layer performs an operation called a “convolution“.

## Recurrent Neural Networks
The idea behind RNNs is to make use of sequential information. In a traditional neural network we assume that all inputs (and outputs) are independent of each other. But for many tasks that’s a very bad idea. If you want to predict the next word in a sentence you better know which words came before it. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations and you already know that they have a “memory” which captures information about what has been calculated so far.

## When can LSTM be used?
Data collected over successive periods of time are characterised as a Time Series. In such cases, an interesting approach is to use a model based on LSTM (Long Short Term Memory), a Recurrent Neural Network architecture. In this kind of architecture, the model passes the previous hidden state to the next step of the sequence. Therefore holding information on previous data the network has seen before and using it to make decisions. In other words, the data order is extremely important.

## When can Convolution be used?
When working with images, the best approach is a CNN (Convolutional Neural Network) architecture. The image passes through Convolutional Layers, in which several filters extract important features. After passing some convolutional layers in sequence, the output is connected to a fully-connected Dense network.

## When can ConvLSTM be used?
For sequencial images, one approach is using ConvLSTM layers. It is a Recurrent layer, just like the LSTM, but internal matrix multiplications are exchanged with convolution operations. As a result, the data that flows through the ConvLSTM cells keeps the input dimension (3D in our case) instead of being just a 1D vector with features.

## Dataset
The dataset that is made available in HARDataset folder is the accelerometer data. 2.56 seconds of timeframe is considered as one window. Each window has 128 time steps. 

## Understanding Epochs and Batch size
### Batch
The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.

Think of a batch as a for-loop iterating over one or more samples and making predictions. At the end of the batch, the predictions are compared to the expected output variables and an error is calculated. From this error, the update algorithm is used to improve the model, e.g. move down along the error gradient.
### Epoch
One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters. An epoch is comprised of one or more batches. For example, as above, an epoch that has one batch is called the batch gradient descent learning algorithm.

You can think of a for-loop over the number of epochs where each loop proceeds over the training dataset. Within this for-loop is another nested for-loop that iterates over each batch of samples, where one batch has the specified “batch size” number of samples.

The number of epochs is traditionally large, often hundreds or thousands, allowing the learning algorithm to run until the error from the model has been sufficiently minimized. You may see examples of the number of epochs in the literature and in tutorials set to 10, 100, 500, 1000, and larger.

## LSTM Neural Network
In this network, each window has 128 time steps. The whole dataset was passed through 15 times. The number of times the experiment has to be repeated can be modified by modifying 
```run_experiment(repeats=1)```

To run this script use: ```python3 run_experiment_lstm.py```

## Neural Network with LSTM and Convolution Layers
In this network, each window is further divided into 4 sub-windows each with 32 time steps. The whole dataset was passed through 25 times. The number of times the experiment has to be repeated can be modified by modifying 
```run_experiment(repeats=1)```

To run this script use: ```python3 run_experiment_lstm_cnn.py```

## ConvlLSTM Neural Network
This networ is designed similar to the earlier one. In this network as well, each window is further divided into 4 sub-windows each with 32 time steps. The whole dataset was passed through 25 times. The number of times the experiment has to be repeated can be modified by modifying 
```run_experiment(repeats=1)```

To run this script use: ```python3 run_experiment_convlstm2d.py```

## Installing Depedencies
Run the below command before running any of the scripts
```
sudo chmod a+x install_dependencies.sh
sudo ./install_dependencies.sh
```

## References
1. https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7
2. https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
3. https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
4. https://hackernoon.com/deep-learning-vs-machine-learning-a-simple-explanation-47405b3eef08
5. https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
