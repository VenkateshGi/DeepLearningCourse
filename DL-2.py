import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = tf.keras.datasets.fashion_mnist

(train_x,train_y),(test_x,test_y) = mnist.load_data()

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
#Step1

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
                                    
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units = 100, activation = tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation = tf.nn.softmax)])                                    
#step2

model.compile(optimizer = "sgd", loss = "cross-entropy-loss")
call_back = myCallback()
#step3
model.fit(train_x, train_y, epochs = 5, callbacks = [callback])

#step4

model.predict(test_x)
