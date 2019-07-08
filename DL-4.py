import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.fashion_mnist

(train_x,train_y),(test_x,test_y) = mnist.load_data()

#since using CNN, we have to reshape the data

train_x = train_x.reshape(60000,28,28,1)
test_x = test_x.reshape(10000,28,28,1)

train_x/=255
test_x/=255
#step1

model = tf.keras.Models.Sequential([tf.keras.Layers.Conv2D(60,(3,3), input_shape = (28,28,1), activation = 'relu'),
                                    tf.keras.Layers.MaxPooling2D(2,2),
                                    tf.keras.Layers.Conv2D(60,(3,3), activation = 'relu'),
                                    tf.keras.Layers.MaxPooling2D(2,2),
                                    tf.keras.Layers.Dense(250, activation = 'relu'),
                                    tf.keras.Layers.Dense(10, activation = 'softmax')])
#ster 2

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#step 3

model.fit(train_x, train_y, epochs = 5)

test_loss = model.evaluate(test_images, test_labels)
