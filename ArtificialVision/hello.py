#Artificial Vision by Matt@CEFET-MG
#Based on DeepFace and FaceNet implementations
#Special Thanks to Deeplearning.ai@Coursera

#MIT License

#Copyright (c) 2018 Deeplearning.ai, 2016 David Sandberg

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#Disclaimer: This code may contain code that could be used to complete the Coursera
#courses this code is partly based on (as has the original facenet and OpenFace-Keras implementations).
#The code was altered to contain as least information that could be useful as possible,
#however no student shall use this code or parts of it to complete their assignements
#in the coursera platform, since plagiarism goes directly against the coursera honour code.


import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from keras.models import Sequential, load_model
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
    Implementation of the triplet loss as defined by Facenet.

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

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss,0))

    return loss

sys.path.append("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/")

pathy = "C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/"

#Initializes a model, compile it optimizing via adam and then proceeds to calculate the loss
FRmodel = faceRecoModel(input_shape=(3,96,96))
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
#FRmodel = load_model(pathy+'FromAndrewNg/nn4.small2.v7.h5')
#Loads the weights from the OpenFace-keras implementation
#load_weights_from_FaceNet(FRmodel)

#%%

database = {}
database["danielle"] = img_to_encoding("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/andrew.jpg", FRmodel)
database["matt"] = img_to_encoding("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/ArtificialVision/images/matt.png", FRmodel)
database["kian"] = img_to_encoding(pathy+"images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding(pathy+"images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding(pathy+"images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding(pathy+"images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding(pathy+"images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding(pathy+"images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding(pathy+"images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding(pathy+"images/arnaud.jpg", FRmodel)
database["alex"] = img_to_encoding(pathy + "images/alexlabossiere.png", FRmodel)
database["matt2"] =img_to_encoding(pathy + "images/matt2.jpg")

def wer_ist_das(image_path, database, model):
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
    min_dist = 100 #high value initializer

    # Loop over the database dictionary's names and encodings.
    for (name) in database.keys():

        # Compute L2 distance between the target "encoding" and the current db_enc from the database.
        dist = np.linalg.norm(encoding-database[name])

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 1.0: #settable parameter for safe recognition
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

#%% Testing cell

wer_ist_das(pathy + "images/test_set/M_test_two.png", database, FRmodel)
#%%
