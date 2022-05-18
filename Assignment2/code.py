from pydoc import describe
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt 
import random





FLOWERS = {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4}

def load_data(base_directory):
    x = []
    y = []
    # numpy.array don't like images with different dimensions so...
    desired_image_size = (100,100)

    flower_dirs = os.listdir(base_directory)
    for flower_dir in flower_dirs:

        # Get each flower dir. E.g. 'small_flower_dataset/daisy/'
        flower_base_dir = os.path.join(base_directory, flower_dir)
        # Get each images full path
        files = glob.glob(os.path.join(flower_base_dir, '*.jpg'))
        print('Flower base directory: '+flower_base_dir)
        
        # Get image data
        for f in files:
            image = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) / 255.0
            image = cv2.resize(image,desired_image_size)
            x.append(image)
            y.append(FLOWERS[flower_dir])
        # break

    return np.array(x), np.array(y)


def plot_images(x, y):
    fig = plt.figure(figsize=[15, 15])
    for i in range(50):
        ax = fig.add_subplot(5, 10, i + 1)
        index = random.randint(0,x.shape[0])
        ax.imshow(x[index,:])
        ax.set_title(y[index])
        ax.axis('off')
    plt.plot()
    plt.savefig('samples')

def task1():
    # flower dataset is located in same directory

    # Load dataset
    
    x, y = load_data('small_flower_dataset')
    plot_images(x,y)
    return x, y

def task2():
    # Using the tf.keras.applications module download a pretrained MobileNetV2
    # network.
    pass

def task3():
    # Replace the last layer with a Dense layer of the appropriate shape given
    # that there are 5 classes in the small flower dataset.
    pass

def task4():
    # Prepare your training, validation and test sets.
    pass

def task5():
    # Compile and train your model with an SGD3 optimizer using the following
    # parameters learning_rate=0.01, momentum=0.0, nesterov=False.
    pass

def task6():
    # Plot the training and validation errors vs time as well as the training
    # and validation accuracies.
    pass

def task7():
    # Experiment with 3 different orders of magnitude for the learning rate.
    # Plot the results, draw conclusions.
    pass

def task8():
    # With the best learning rate that you found in the previous task, add a
    # non zero momentum to the training with the SGD optimizer (consider 3
    # values for the momentum). Report how your results change. 
    pass

if __name__ == "__main__":
    task1()
    # task2()
    # task3()
    # task4()
    # task5()
    # task6()
    # task7()
    # task8()