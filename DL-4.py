import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.fashion_mnist

(train_x,train_y),(test_x,test_y) = mnist.load_data()

#since using CNN, we have to reshape the data

train_x = train_x.reshape(60000,28,28,1)
test_x = test_x.reshape(10000,28,28,1)

train_x/=255.0
test_x/=255.0

#step1

model = tf.keras.models.Sequential
