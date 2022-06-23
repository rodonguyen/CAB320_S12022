# %%
'''
CAB320 Assignment 2 Submission
Due: 12th June 2022

Group Members:
    Mitchell Egan     n10210776
    Jaydon Gunzburg   n10396489
    Rodo Nguyen       n10603280
    
We have written this program to be run in blocks. First Block contains all the
functions and imports that are used, with the remaining corresponding to each 
task of the assignment.
The flower dataset folder 'small_flower_dataset' must be in the same directory.
Models and figures are both saved to the 'figures' folder.
'''


# neural network
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, smart_resize
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# loading and saving data
import os
import glob
import pickle
# data plotting and manipulation
import numpy as np
import matplotlib.pyplot as plt 
import random

# Global variables
FLOWERS = {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4}
DESIRED_IMAGE_SIZE = (224,224)
INPUT_SHAPE = DESIRED_IMAGE_SIZE + (3,)
OUTPUT_NUMBER_2ND_LAST_DENSE = 2^6
NUMBER_OF_CLASSES = len(FLOWERS)
run_num = '0' # For data saving purpose

# Increase text size of the plots
plt.rcParams.update({'font.size': 20})

# Create /figures directory if not exist
import os
path = "figures"
if (not os.path.exists(path)):
    os.makedirs(path)
    print("The /figures directory is created!")



def save_data(filename, data):
    '''
    This function is used to save important data without having to 
    rerun the program to get it.

    Parameters
    ----------
    filename: string,
        the filename which data will be saved to
    data: object
        value / variable
    '''
    with open(f"{filename}", "wb") as f:
        pickle.dump(data, f)


def load_dataset(base_flower_directory = 'small_flower_dataset'):
    '''
    loads the dataset contained in 'small_flower_dataset' 
    and scales pixel values to [0,1] value range
    
    Parameters
    ----------
    base_flower_directory: string
        the path to small_flower_dataset

    Returns
    -------
    x: an array of data
    y: an array of labels
    '''
    x = [] # images
    y = [] # respective image label
    for flower_dir in os.listdir(base_flower_directory):
        flower_dir_full = os.path.join(base_flower_directory, flower_dir)
        flower_files = glob.glob(os.path.join(flower_dir_full, '*.jpg'))
        print('Loading flower directory: ' + flower_dir_full)
        for f in flower_files:
            image = load_img(f)
            # resize all images to the same predefined size
            image = smart_resize(image, DESIRED_IMAGE_SIZE)
            x.append(image)
            y.append(FLOWERS[flower_dir])

    # Scale pixel values to [0,1] range
    return np.array(x)/255, np.array(y)


def plot_images(x, y):
    '''
    Display 50 of the images (used in debugging)
    
    Parameters
    ----------
    x: array
        Data
    y: array
        Labels of the data (x)
    '''
    fig = plt.figure(figsize=[15, 15])
    for i in range(50):
        ax = fig.add_subplot(5, 10, i + 1)
        index = random.randint(0,x.shape[0]-1)
        ax.imshow(x[index,:])
        ax.set_title(y[index])
        ax.axis('off')
    plt.plot()
    plt.savefig('samples')


def plot_data_distribution(y_train, y_val, y_test):
    '''
    Plots data distrbution of the training, validation and test sets   
    
    Parameters
    ----------
    y_train, y_val, y_test: array
        Labels of the data
    '''
    y_train_order = np.sort(y_train)
    y_val_order = np.sort(y_val)
    y_test_order = np.sort(y_test)

    fig = plt.figure(figsize=[18,8])
    ax = fig.add_subplot(1, 3, 1)
    y_train_hist = ax.hist(y_train_order,5)
    print(y_train_hist[0])
    ax.set_title("Train data")

    ax = fig.add_subplot(1, 3, 2)
    y_val_hist = ax.hist(y_val_order,5)
    print(y_val_hist[0])
    ax.set_title("Val data")

    ax = fig.add_subplot(1, 3, 3)
    y_test_hist = ax.hist(y_test_order,5)
    print(y_test_hist[0])
    ax.set_title("Test data")

    plt.savefig('figures/Data_distribution')
    

def download_model():
    '''
    Using the tf.keras.applications module download a pretrained MobileNetV2
    with all layers set to untrainable    

    Returns
    -------
    model: a Keras Model object
    '''
    model = MobileNetV2(
                input_shape = INPUT_SHAPE,
                include_top = False,
                weights = 'imagenet')
    
    # Freeze all layers    
    for layer in model.layers:
        layer.trainable = False

    return model


def replace_model_layer(model):
    '''
    Replace the output layer of 'model'
    Set final Dense layer's output to 5 - the number of flower classes to classify

    Parameters
    ----------
    model: a Keras Model object

    Returns
    -------
    new_model: a Keras Model object with replaced output layer
    '''
    x = Flatten()(model.layers[-1].output)
    # 2^6 is the number between its input and the required output
    # The Dense layer below acts as an intermediate layer to
    # better aggregate learned features
    x = Dense(OUTPUT_NUMBER_2ND_LAST_DENSE, activation='relu')(x)  
    x = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)

    new_model = Model(inputs=model.layers[0].input,
                      outputs = x,
                      name = 'mobilenetv2_modified')
    
    return new_model


def create_untrained_model():
    '''
    Returns a mobilenetv2 network with the last layer replaced

    Returns
    -------
    a Keras Model object with replaced output layer
    '''
    return replace_model_layer(download_model())


def train_model(model, LR, M, run_num, x_train, x_val, y_train, y_val):  
    '''
    Trains the input model.
    After the performance converges, unfreeze all layers 
    and train again.

    Parameters
    ----------
    model: a Keras Model object
        Model to be trained
    LR: float
        Learning rate
    M: float
        Momentum
    run_num: string
        The run number to be inserted to filenames
    x_train: array 
        Train data
    x_val: array
        Validation data
    y_train: array
        Train labels
    y_val: array
        Validation labels

    Returns
    -------
    model: a Keras Model object with replaced output layer
    history_freeze: History of training partial frozen model 
    history_unfreeze: History of training unfrozen model
    '''

    # Define variables / parameters
    monitor = 'val_loss'
    epochs = 300
    batch_size = 50
    
    # Define filenames to save
    savefilename = f"figures/model_runNumber{run_num}_{LR}_{M}"
    savefilename_1 = savefilename + "_freezeSomeLayers_highestAccuracy.h5"
    savefilename_2 = savefilename + "_unfreezeAllLayers_lowestLoss.h5"
    savefilename_3 = savefilename + "_unfreezeAllLayers_highestAccuracy.h5"
    savefilename_history_1 = savefilename+"_freezeSomeLayers_history"
    savefilename_history_2 = savefilename+"_unfreezeAllLayers_history"

    # Assistive callbacks
    # Stop the training if validation_accuracy hasn't improved
    earlyCallback = EarlyStopping(monitor = monitor,
                                  patience = 20)
    # Save model when val_accuracy improves
    modelCheckpoint = ModelCheckpoint(savefilename_1, 
                                      verbose=1,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='auto')

    # Compile
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizers.SGD(learning_rate=LR,
                                                 momentum=M,
                                                 nesterov=False),
                  metrics=['accuracy'])

    # Train the model with some layers frozen
    history_freeze = model.fit(x_train, y_train,
                               batch_size = batch_size,
                               epochs = epochs,
                               validation_data = (x_val, y_val),
                               callbacks = [earlyCallback, modelCheckpoint])
    
    save_data(savefilename_history_1, history_freeze.history)

    # Continue to train the model with all layers unfrozen
    for layer in model.layers:
        layer.trainable = True

    # Load params from the best saved model and continue from there
    model.load_weights(savefilename_1)

    # Assistive callbacks
    earlyCallback = EarlyStopping(monitor = monitor,
                                  patience = 20)
    modelCheckpointLoss = ModelCheckpoint(savefilename_2, 
                                          verbose=1,
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='auto')
    modelCheckpointAccuracy = ModelCheckpoint(savefilename_3, 
                                              verbose=1,
                                              monitor='val_accuracy',
                                              save_best_only=True,
                                              mode='auto')
    
    # Compile
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizers.SGD(learning_rate=LR,
                                               momentum=M,
                                               nesterov=False),
                  metrics=['accuracy'])

    # Train the unfreezed model
    history_unfreeze = model.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(x_val, y_val),
                                 callbacks=[earlyCallback,
                                            modelCheckpointLoss,
                                            modelCheckpointAccuracy])
    
    save_data(savefilename_history_2, history_unfreeze.history)
    model.load_weights(savefilename_2)

    return model, history_freeze, history_unfreeze


def get_test_accuracy(model, x_test, y_test):
    '''
    Prints and saves (to 'figures' folder) the accuracies of the input model

    Parameters
    ----------
    model: a Keras Model object
    '''
    filename1 = f"figures/model_runNumber{run_num}_{LR}_{M}_unfreezeAllLayers_lowestLoss.h5"
    model.load_weights(filename1)
    pred = model.predict(x_test)
    indexes = tensorflow.argmax(pred, axis=1)
    accuracy1 = np.sum(indexes == y_test[:]) / len(y_test)
    print('Test Accuracy from lowest Loss model:', accuracy1) 

    filename2 = f"figures/model_runNumber{run_num}_{LR}_{M}_unfreezeAllLayers_highestAccuracy.h5"
    model.load_weights(filename2)
    pred = model.predict(x_test)
    indexes = tensorflow.argmax(pred, axis=1)
    accuracy2 = np.sum(indexes == y_test[:]) / len(y_test)
    print('Test Accuracy from highest Accuracy model:', accuracy2) 

    filename3 = f"figures/model_runNumber{run_num}_{LR}_{M}_freezeSomeLayers_highestAccuracy.h5"
    model.load_weights(filename3)
    pred = model.predict(x_test)
    indexes = tensorflow.argmax(pred, axis=1)
    accuracy3 = np.sum(indexes == y_test[:]) / len(y_test)
    print('Test Accuracy from model before unfreezing all layers:', accuracy3) 

    result_file = open('test_results.csv', 'a')
    result_file.write(f"{filename1}, {accuracy1}\n")
    result_file.write(f"{filename2}, {accuracy2}\n")
    result_file.write(f"{filename3}, {accuracy3}\n\n")
    result_file.close()


def plot_training_history(history, savefilename):
    '''
    Plot the training and validation errors vs time as well as 
    the training and validation accuracies.

    Parameters
    ----------
    history: history 
        History from fitting model
    savefilename: string
        The file name to save plot
    '''
    fig = plt.figure(figsize=[20, 20])
    
    # Train and validation loss
    ax = fig.add_subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_title('Loss vs Epoch')

    # Training and validation accuracy
    ax = fig.add_subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_title('Accuracy vs Epoch')

    plt.savefig(savefilename)

    # Print the best loss and accuracy
    print('Min loss: ' + str(min(history.history['loss'])))
    print('Max accuracy: ' + str(max(history.history['accuracy'])))
    print('Min loss_accuracy: ' + str(min(history.history['val_loss'])))
    print('Max val_accuracy: ' + str(max(history.history['val_accuracy'])))


# %%
###############################################################################
#                               Run Script                                    #
###############################################################################
# UPDATE RUN NUMBER EACH TIME SCRIPT IS USED, for data saving purpose
run_num = '77'

# %% Task 1 - Loading the data
x, y = load_dataset(base_flower_directory = 'small_flower_dataset')

#print('Whole dataset:', x.shape, y.shape)
# plot_images(x, y)


# %% Task 2 - Download the pretrained network
model = download_model()

# %% Task 3 - Replacing the last layer
model = replace_model_layer(model)
# print the model summary
# model.summary()


# %% Task 4 - Splitting the data
x_train, x_valTest, y_train, y_val_test = \
    train_test_split(x, y, test_size=0.3, random_state=42)

x_val, x_test, y_val, y_test = \
    train_test_split(x_valTest, y_val_test, test_size=0.5, random_state=42)

### Check each set's shape
# print(x_train.shape, '\t', y_train.shape)
# print(x_val.shape, '\t', y_val.shape)
# print(x_test.shape, '\t', y_test.shape)
### Check data distribution
# plot_data_distribution(y_train, y_val, y_test)


# %% Task 5 - Compiling and Training the model
LR = 0.01
M = 0.0
batch_size = 50
modelLR01, hist_freeze_LR01, hist_unfreeze_LR01 = \
    train_model(create_untrained_model(), LR, M, run_num,
                x_train, x_val, y_train, y_val)

# Get Test Accuracy
get_test_accuracy(modelLR01, x_test, y_test)

# %% Task 6 - Plot training history
plot_training_history(hist_freeze_LR01,
            f"figures/model_runNumber{run_num}_001_0_freezeSomeLayers_history")
plot_training_history(hist_unfreeze_LR01,
            f"figures/model_runNumber{run_num}_001_0_unfreezeAllLayers_history")


# %% Task 7 - Compiling with 3 learning rates
M = 0.0
LR = 0.1
model_0701, hist_freeze_0701, hist_unfreeze_0701 = \
            train_model(create_untrained_model(), LR, M, run_num,
            x_train, x_val, y_train, y_val)
plot_training_history(hist_freeze_0701,
            f"figures/model_runNumber{run_num}_01_0_freezeSomeLayers_history")
plot_training_history(hist_unfreeze_0701,
            f"figures/model_runNumber{run_num}_01_0_unfreezeAllLayers_history")
get_test_accuracy(model_0701, x_test, y_test)

# ---

LR = 0.001
model_0702, hist_freeze_0702, hist_unfreeze_0702 = \
            train_model(create_untrained_model(), LR, M, run_num,
            x_train, x_val, y_train, y_val)
plot_training_history(hist_freeze_0702,
            f"figures/model_runNumber{run_num}_0001_0_freezeSomeLayers_history")
plot_training_history(hist_unfreeze_0702,
            f"figures/model_runNumber{run_num}_0001_0_unfreezeAllLayers_history")
get_test_accuracy(model_0702, x_test, y_test)

# ---

LR = 0.0001
model_0703, hist_freeze_0703, hist_unfreeze_0703 = \
            train_model(create_untrained_model(), LR, M, run_num,
            x_train, x_val, y_train, y_val)
plot_training_history(hist_freeze_0703,
            f"figures/model_runNumber{run_num}_00001_0_freezeSomeLayers_history")
plot_training_history(hist_unfreeze_0703,
            f"figures/model_runNumber{run_num}_00001_0_unfreezeAllLayers_history")
get_test_accuracy(model_0703, x_test, y_test)



# %% Task 8 - Choose 3 momentum values
LR = 0.01  # Assign the best LR from Task 5 and 7
M = 0.01 # low but not too insignifcant momentum
model_0801, hist_freeze_0801, hist_unfreeze_0801 = \
            train_model(create_untrained_model(), LR, M, run_num,
            x_train, x_val, y_train, y_val)
plot_training_history(hist_freeze_0801,
            f"figures/model_runNumber{run_num}_001_005_freezeSomeLayers_history")
plot_training_history(hist_unfreeze_0801,
            f"figures/model_runNumber{run_num}_001_005_unfreezeAllLayers_history")
get_test_accuracy(model_0801, x_test, y_test)


M = 0.5 # medium value 
model_0802, hist_freeze_0802, hist_unfreeze_0802 = \
            train_model(create_untrained_model(), LR, M, run_num,
            x_train, x_val, y_train, y_val)
plot_training_history(hist_freeze_0802,
            f"figures/model_runNumber{run_num}_001_05_freezeSomeLayers_history")
plot_training_history(hist_unfreeze_0802,
            f"figures/model_runNumber{run_num}_001_05_unfreezeAllLayers_history")
get_test_accuracy(model_0802, x_test, y_test)


M = 1 # max momentum
model_0803, hist_freeze_0803, hist_unfreeze_0803 = \
            train_model(create_untrained_model(), LR, M, run_num,
            x_train, x_val, y_train, y_val)
plot_training_history(hist_freeze_0803,
            f"figures/model_runNumber{run_num}_001_1_freezeSomeLayers_history")
plot_training_history(hist_unfreeze_0803,
            f"figures/model_runNumber{run_num}_001_1_unfreezeAllLayers_history")
get_test_accuracy(model_0803, x_test, y_test)



