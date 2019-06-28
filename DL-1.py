#import necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np

#basic steps for building a model
#1. Build the Network
#2. Define the Compile Model
#3. Fit the model with the Training Data
#4. Predict the results of Test Data

#1
model = tf.keras.Sequential([Keras.layers.Dense(units = 1, input_shape = [1])])

#2
model.compile(optimizer = "sgd", loss = 'mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#3
model.fit(xs, ys, epochs = 500)

#4
print(model.predict([10.0]))
