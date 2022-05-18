# Python is 3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make the output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# MNIST


# EXERCISE 1 

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

# list the keys
print(mnist.keys())

# mnist["data"] corresponds to the input examples
# mnist["target"] are the class labels of the examples
X, y = mnist["data"], mnist["target"]
print('Shape of X and y ',X.shape, y.shape)

if False: # change to True to execute the code
#if True: # change to True to execute the code
    some_digit = X[99]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")    
    save_fig("some_digit_plot")
    plt.show()


# EXERCISE 2

# type of the entries of y
# y is numpy array of str

if False: # change to True to execute the code
#if True: # change to True to execute the code

    # convert to uint8 integer
    y = y.astype(np.uint8)
     
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    
    
    # Build the linear classifier
    from sklearn.linear_model import SGDClassifier
    
    sgd_clf = SGDClassifier()
    
    # train the model
    sgd_clf.fit(X_train, y_train_5)
    
    
    # Use sgd_clf to make predictions
    y_predicted = sgd_clf.predict(X_test[50:80])
    
    # you can then re-use the code from EXERCISE 1 to explore the results
    i=57 # an index in X_test
    some_digit = X_test[i]
    
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.title('Predicted is 5 : '+str(y_predicted[i-50]))
    plt.axis("off")
    #save_fig("some_digit_plot")
    plt.show()

    # Apply cross validation
    from sklearn.model_selection import cross_val_score
    print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


# EXERCISE 3

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

if False: # change to True to execute the code
#if True: # change to True to execute the code
    
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    
    print(confusion_matrix(y_train_5, y_train_pred))
    
    #    Each row in a confusion matrix represents an actual class, 
    #    while each column represents a predicted class. 
    #    The first row of this matrix considers non-5 images (the negative class)


# EXERCISE 4, 

from sklearn.metrics import precision_score, recall_score
#if True: # change to True to execute the code
if False: # change to True to execute the code
    print(precision_score(y_train_5, y_train_pred))
    print(recall_score(y_train_5, y_train_pred))

# EXERCISE 5
from sklearn.ensemble import RandomForestClassifier
if False: # change to True to execute the code
#if True: # change to True to execute the code
    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    forest_clf.fit(X_train, y_train)
    y_test_pred = forest_clf.predict(X_test)
    print(confusion_matrix(y_test, y_test_pred))

# EXERCISE 6

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

#if False: # change to True to execute the code
if True: # change to True to execute the code    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    iterations=1000   # define the iterations for training over the dataset
    hidden_layers=[10,10,10]  # define the layers/depth of the NN
    print("Creating a neural network with "+str(len(hidden_layers))+
          " layers and "+str(iterations)+" iterations")
    
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, verbose=True, max_iter=iterations) 
    
    # an object which represents the neural network
    # Remember to use the pre-processed data and not original values for fit()
    
    mlp.fit(X_train_scaled, y_train)  # fit features over NN
    
    ## Run the test data over the network to see the predicted outcomes.
    y_test_pred = mlp.predict(X_test_scaled)
    print(confusion_matrix(y_test, y_test_pred))
