# setup
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# prepare the data

# model and data parameters
num_classes = 10
input_shape = (28*28,)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Flatten the images to 1D vectors 
x_train = x_train.reshape((x_train.shape[0],-1))
x_test = x_test.reshape((x_test.shape[0],-1))

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build the model

model = keras.Sequential(
    [

# INSERT YOUR CODE HERE 

    ]
)

model.summary()


# Train the model

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(
        
# INSERT YOUR CODE HERE 

        epochs=epochs, 
        validation_split=0.1)

# Evaluate the trained model

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

