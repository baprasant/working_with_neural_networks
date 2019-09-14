# sudo chmod a+x install_dependencies.sh
# ./install_dependencies.sh

# Install main dependencies
sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv  # system-wide install

# Install Numpy for working with arrays and vectors
sudo pip3 install numpy

# Install pandas for getting data from .csv files and text files
sudo pip3 install pandas

# Install Keras for using neural networks
sudo pip3 install Keras

# Install Tensor Flow
sudo pip3 install tensorflow

# Verify the install
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

# Install matplotlib
sudo pip3 install matplotlib
