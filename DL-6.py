#cats vs dogs
import tensorflow as tf

#step 1

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), input_shape = (300, 300, 3), activation = 'relu'),
                                    tf.keras.layers.Maxpooling2D(2,2),
                                    tf.keras.layers.Conv2D(32,(3,3), activation = "relu"),
                                    tf.keras.layers.Maxpooling2D(2,2),
                                    tf.keras.layers.Conv2D(32,(3,3), activation = "relu"),
                                    tf.keras.layers.Maxpooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    tf.keras.layers.Dense(1, activation = 'sigmoid')
                                    
                                    ])
