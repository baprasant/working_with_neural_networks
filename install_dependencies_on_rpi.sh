#!/bin/bash

# Update
sudo apt-get update

# For python3
sudo apt-get install python3-numpy python3-dev python3-pip python3-mock

# For Keras
pip3 install -U --user keras_applications==1.0.8 --no-deps
pip3 install -U --user keras_preprocessing==1.1.0 --no-deps

# For Tensorflow
pip3 install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.1.0/tensorflow-2.1.0-cp37-none-linux_armv7l.whl

# Checking Installation
python3 -c "import tensorflow as tf;print(tf.__version__)"
# Should be getting 2.1.0

pip3 install pandas
pip3 install keras==2.3.1
pip3 install matplotlib


sudo apt-get install libopenjp2-7
sudo apt-get install libtiff5
