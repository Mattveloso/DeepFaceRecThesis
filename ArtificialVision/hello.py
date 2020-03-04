#Artificial Vision by Matt@CEFET-MG
#Based on DeepFace and FaceNet implementations
#Special Thanks to Deeplearning.ai@Coursera

import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import sys
sys.path.append("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/")
sys.path.append("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/FromAndrewNg/")
from fr_utils import *
from inception_blocks_v2 import *
import utils
from utils import LRN2D

%matplotlib inline
%load_ext autoreload
%autoreload 2

np.set_printoptions(threshold=sys.maxsize)

# %% This breaks the code into cells. Here is the beginning of the code ------------
#This is a creation of a model in Keras, see inception_blocks_v2

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by Ng.

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Computes the (encoding) distance between the anchor, the positive and the negative
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)
    # Subtracts the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    # Takes the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss,0))

    return loss

#Initializes a model, compile it optimiying via adam and then proceeds to calculate the loss
FRmodel = faceRecoModel(input_shape=(3,96,96)) # Understand this model better
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

#Loads the weights from the OpenFace-keras implementation
load_weights_from_FaceNet(FRmodel)

#%%
sys.path.append("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/")

database = {}
database["danielle"] = img_to_encoding("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/andrew.jpg", FRmodel)
database["matt"] = img_to_encoding("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/matt.png", FRmodel)

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    # Computes the target "encoding" for the image.
    encoding = img_to_encoding(image_path,model)

    # Finds the closest encoding ##
    min_dist =  #high value initializer

    # Loop over the database dictionary's names and encodings.
    for (name) in database.keys():

        # Compute L2 distance between the target "encoding" and the current db_enc from the database.
        dist = np.linalg.norm(encoding-database[name])

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7: #settable parameter for safe recognition
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

#%%

who_is_it("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/teste_1.jpg", database, FRmodel)
#%%
who_is_it("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/camera_0.jpg", database, FRmodel)
