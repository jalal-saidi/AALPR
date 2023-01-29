# coding: utf-8

# Author Said Jalal Saidi
# Training the Farsi character OCR model using Keras


import os

import keras
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import numpy as np
# From pypng
import png
from keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential

# from sklearn.p

FONTSIZE = 18
FIGURE_SIZE = (10, 4)
FIGURE_SIZE2 = (10, 10)

# Configure parameters
plt.rcParams.update({'font.size': FONTSIZE})

####################################
# 
import re

##################
# Default tick label size
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24


# from hw4_tools import *


# In[5]:


def readPngFile(filename):
	'''
    Read a single PNG file
    
    filename = fully qualified file name
    
    Return: 3D numpy array (rows x cols x chans)
    
    Note: all pixel values are floats in the range 0.0 .. 1.0
    
    This implementation relies on the pypng package
    '''

	# Load in the image meta-data
	r = png.Reader(filename)
	it = r.read()

	# Load in the image itself and convert to a 2D array
	image_2d = np.vstack(map(np.uint8, it[2]))

	# Reshape into rows x cols x chans
	image_3d = np.reshape(image_2d,
						  (it[0], it[1], it[3]['planes'])) / 255.0

	return image_3d


def read_images_from_directory(directory, file_regexp):
	'''
    Read a set of images from a directory.  All of the images must be the same size
    
    directory = Directory to search
    
    file_regexp = a regular expression to match the file names against
    
    Return: 4D numpy array (images x rows x cols x chans)
    '''

	# Get all of the file names
	files = sorted(os.listdir(directory))

	# Construct a list of images from those that match the regexp
	# list_of_images = [readPngFile(directory+'/'+f) for f in files if re.search(file_regexp, f)]

	list_of_images = []
	list_of_classes = []
	for f in files:
		if re.search(file_regexp, f):

			res = readPngFile(directory + '/' + f)

			list_of_images.append(np.array(res))
			match = re.search('_(\d*).png', f)
			if match:
				list_of_classes.append(float(match.group(1)))

	# Create a 3D numpy array
	return (list_of_images, list_of_classes)


# In[7]:


directory = './resized'
result = read_images_from_directory(directory, 'numplate')
images = np.array(result[0], dtype=np.float32)
classes = np.array(result[1], dtype=np.float32)

# In[8]:


sh = np.arange(images.shape[0])
np.random.shuffle(sh)
ins = images[sh]
outs = classes[sh]

# In[14]:


train_size = 4000
valid_size = 2000

X_train = ins[:train_size]
y_train = outs[:train_size]

X_valid = ins[train_size:train_size + valid_size]
y_valid = outs[train_size:train_size + valid_size]

X_test = ins[train_size + valid_size:]
y_test = outs[train_size + valid_size:]

num_category = 62
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_valid = keras.utils.to_categorical(y_valid, num_category)

y_test = keras.utils.to_categorical(y_test, num_category)


# In[10]:


# In[11]:


def model_ocr(X_train, y_train, X_valid, y_valid, num_epoch, batch_size):
	model = Sequential()
	lam = 0.0
	kernel_regularizer = keras.regularizers.l2(lam)
	bias_regularizer = keras.regularizers.l2(lam)

	model.add(Convolution2D(32, (3, 3), activation='relu', strides=1
							, input_shape=(30, 30, 3), kernel_regularizer=kernel_regularizer,
							bias_regularizer=bias_regularizer, kernel_initializer='random_uniform',
							bias_initializer='random_uniform', name='C0'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='M0'))
	model.add(Convolution2D(64, (3, 3), strides=1, activation='relu', kernel_regularizer=kernel_regularizer,
							bias_regularizer=bias_regularizer, kernel_initializer='random_uniform',
							bias_initializer='random_uniform', name='C1'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='M1'))
	model.add(Convolution2D(128, (3, 3), strides=1, activation='relu', kernel_regularizer=kernel_regularizer,
							bias_regularizer=bias_regularizer, kernel_initializer='random_uniform',
							bias_initializer='random_uniform', name='C3'))
	model.add(MaxPooling2D((2, 2), name='M3'))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
					kernel_initializer='random_uniform', bias_initializer='random_uniform'))
	model.add(Dense(128, activation='relu', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
					kernel_initializer='random_uniform', bias_initializer='random_uniform'))
	# model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(62, activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform'))

	model.compile(optimizer=keras.optimizers.Adadelta(),
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])
	model_log = None

	return (model, model_log)
	# model_log = model.fit(X_train, y_train,
	#       batch_size=batch_size,
	#       epochs=num_epoch,
	#       verbose=1,
	#       validation_data=(X_valid, y_valid))
	model_log = None



# In[15]:


num_epoch = 50
batch_size = 128

model, model_log = model_ocr(X_train, y_train, X_valid, y_valid, num_epoch, batch_size)

# In[16]:


# scoreTest = model.evaluate(X_test, y_test, verbose=0)
# print('Test loss:', scoreTest[0])
# print('Test accuracy:', scoreTest[1])


# In[17]:


# scoreValid = model.evaluate(X_valid, y_valid, verbose=0)
# print('valid loss:', scoreValid[0])
# print('valid accuracy:', scoreValid[1])


# In[18]:


# model.save("ALPR_OCR.h5")
model_json = model.to_json()
with open("ALPR_OCR.json", 'w') as json_file:
	json_file.write(model_json)
