import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

# Skipping getting data for now

# VERSION 1: Class-based model
class Model:
    def __init__(self, input_size):
        self.model = tf.keras.Sequential()
        # depth, frame size are first 2 args
        # First layer of a Sequential Model should get input_shape as arg
        # Input: 227 x 227 x 3
        self.model.add(layers.Conv2D(
                12, 
                11, 
                strides=4, 
                activation=activations.relu,
                input_shape=input_size,
        ))
        # Size: 55 x 55 x 12
        self.model.add(layers.MaxPool2D(
                pool_size=3,
                strides=2,
        ))
        # Size: 27 x 27 x 12
        self.model.add(layers.Conv2D(
                18,
                3,
                strides=1,
                activation=activations.relu,
        ))
        # Size: 25 x 25 x 18
        self.model.add(layers.MaxPool2D(
                pool_size=3,
                strides=2,
        ))
        # Size: 12 x 12 x 18
        self.model.add(layers.Flatten())
        # Size: 2592
        self.model.add(layers.Dense(512, activation=activations.relu))
        self.model.add(layers.Dense(128, activation=activations.relu))
        self.model.add(layers.Dense(32, activation=activations.relu))
        # Size of last Dense layer MUST match # of classes
        self.model.add(layers.Dense(5, activation=activations.softmax))
        self.optimizer = optimizers.Adam(learning_rate=0.0001)
        self.loss = losses.CategoricalCrossentropy()
        self.model.compile(
                loss = self.loss,
                optimizer = self.optimizer,
                metrics = ['accuracy'],
        )

model = Model((227, 227, 3))
model.model.summary()