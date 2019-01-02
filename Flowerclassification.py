import numpy as np 
import pandas as pd 
import keras
#import cv2
import scipy
from skimage import io
from PIL import ImageFile
from tqdm import tqdm
import matplotlib.pylab as plt
from matplotlib import cm
from keras.preprocessing import image as keras_image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from bokeh.plotting import figure, show

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
#Plotting
def loss_plot(fit_history):   
    fig_loss = figure(x_axis_label='No. of epochs', y_axis_label='Loss')
    fig_loss.line(range(1,100), fit_history.history['loss'],legend='Training loss',color='red',  line_width=4)
    fig_loss.line(range(1,100), fit_history.history['val_loss'],legend='Validation loss' ,color='blue', line_width=4)
    show(fig_loss)

def acc_plot(fit_history):
    fig_acc = figure(x_axis_label='No. of epochs', y_axis_label='Accuracy')
    fig_acc.line(range(1,100), fit_history.history['acc'],legend='Training accuracy',color = 'red', line_width=4)
    fig_acc.line(range(1,100), fit_history.history['val_acc'],legend='Validation accuracy' ,color='blue', line_width=4)
    show(fig_acc)

#Convert image file into arrays
def get_image_array(path):
    image = keras_image.load_img("flower_images/" + path,target_size=(128, 128))
    arr = keras_image.img_to_array(image)
    exp = np.expand_dims(arr, axis=0)
    return exp

def get_image_array_list(paths):
    list_image_array = [get_image_array(path) for path in tqdm(paths)]
    #stack the image arrays vertically
    return np.vstack(list_image_array)

ImageFile.LOAD_TRUNCATED_IMAGES = True 

file_details = pd.read_csv("flower_images/flower_labels.csv")
file_names = file_details['file']
target_labels = file_details['label'].values

print('Label: ', target_labels[168])

flower_data_array = get_image_array_list(file_names);
print(flower_data_array.shape)

#split flower data into training and testing 80-20%
x_train, x_test, y_train, y_test = train_test_split(flower_data_array, target_labels, test_size = 0.2, random_state = 1)
#Now, further split testing data into testing and validation data 50-50%
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size = 0.5, random_state = 1)

#n = int(len(x_test)/2)
#x_valid, y_valid = x_test[:n], y_test[:n]
#x_test, y_test = x_test[n:], y_test[n:]
print(x_train.shape, x_test.shape, x_validation.shape, y_train.shape, y_test.shape, y_validation.shape)
#
#print('Label: ', y_train[1])
#plt.figure(figsize=(3,3))
#plt.imshow((x_train[1]/255).reshape(128,128,3));
#Can bring this back
#x_train = x_train.astype('float32')/255
#x_test = x_test.astype('float32')/255
#x_validaton = x_validation.astype('float32')/255

#z-score normalization
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std)
x_test = (x_test-mean)/(std)
x_validation = (x_validation-mean)/(std)
#convert target vectors into matrices
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
y_validation_cat = to_categorical(y_validation, 10)

# creating aa Convolutional Neural Network (CNN)
cnn_model = keras.models.Sequential()

cnn_model.add(keras.layers.Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
cnn_model.add(keras.layers.Activation('relu'))

cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(keras.layers.Dropout(0.25))

cnn_model.add(keras.layers.Conv2D(96, (5, 5)))
cnn_model.add(keras.layers.Activation('relu'))

cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(keras.layers.Dropout(0.25))

# model.add(Flatten())
cnn_model.add(GlobalAveragePooling2D())

cnn_model.add(keras.layers.Dense(512, activation='tanh'))
cnn_model.add(keras.layers.Dropout(0.25)) 

# model.add(Dense(256, activation='tanh'))
# model.add(Dropout(0.25)) 

cnn_model.add(keras.layers.Dense(128, activation='tanh'))
cnn_model.add(keras.layers.Dropout(0.25)) 

cnn_model.add(keras.layers.Dense(10))
cnn_model.add(keras.layers.Activation('softmax'))

cnn_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

cnn_mc_history = cnn_model.fit(x_train, y_train_cat, epochs=100, batch_size=64, verbose=2, validation_data=(x_validation, y_validation_cat))
#Plot graph for cnn---is fairly accurate
#repeat for loss as well...and programme is complete
#tweak the layers a bit in both models if possible
#finally, save the model for usage, and show prediction using the model for a sample image
#can also do a comparison for CNN and MLP

loss_plot(cnn_mc_history)
acc_plot(cnn_mc_history)
