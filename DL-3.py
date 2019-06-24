import tensorflow as tf
from tensorflow import keras

#prestep

mnist_digits = tf.keras.datasets.mnist
(train_x, train_y),(test_x, test_y) = mnist_digits.load_data()

class myClass(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

train_x /=255
test_x/=255
#step 1

model = tf.keras.models.Sequential([tf.keras.Layers.Flatten(),
                                    tf.keras.Layers.Dense(150, activation = tf.nn.relu),
                                    tf.keras.Layers.Dense(10, activation = tf.nn.softmax)])
                                   
 #step 2
 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
       
call_back = myClass()       
#step 3
model.fit(train_x, train_y, call_backs = [call_back])

#step4
model.predict(test_x)
