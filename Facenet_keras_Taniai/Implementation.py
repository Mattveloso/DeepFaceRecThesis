from functools import partial

from keras.models import Model
from keras.models import load_model #M
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import add
from keras import backend as K

import sys
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
sys.path.append("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/Facenet_keras_Taniai/code/")
sys.path.append("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/Facenet_keras_Taniai/model/")
sys.path.append("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/Facenet_keras_Taniai/weights/")
sys.path.append("C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/Facenet_keras_Taniai/")

import mtcnn
from mtcnn import MTCNN
import PIL
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from numpy import load
from numpy import expand_dims
import numba
from numba import njit, jit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import sklearn
from random import choice

#Import necessary created functions from inception_resnet_v1.py
from inception_resnet_v1 import scaling
from inception_resnet_v1 import conv2d_bn
from inception_resnet_v1 import _generate_layer_name
from inception_resnet_v1 import _inception_resnet_block
from inception_resnet_v1 import extract_face
from inception_resnet_v1 import load_dataset
from inception_resnet_v1 import get_embedding

# %% Load train Images - M: Slowest part of the code, improve
path1 = "C:/Users/Matt/Documents/GitHub/DeepFaceRecThesis/"
x,y = load_dataset(path1+"Facenet_keras_Taniai/data/images/")
print(x.shape)
print(y.shape)

# %% Load Test images
testx, testy = load_dataset(path1+"Facenet_keras_Taniai/data/Test/")

# %% savez_compressed
savez_compressed('my_dataset.npz', x, y, testx, testy)

# %% Loading and executing
# load the face dataset - for when the npz file is already created previously
data = load('my_dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Carregado: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# load the facenet model
model = load_model(path1+'Facenet_keras_Taniai/model/facenet_keras.h5')
print('Modelo Carregado')

#%%
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
# %%

# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
	embedding = get_embedding(model, face_pixels)
	newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)

# %% Debugging test: l2 distance

for person in newTestX:
    for employee in newTrainX:
        dist = np.linalg.norm(example-employee)#Calculate L2 distance between the two
        print(dist)

# %%


# save arrays to one file in compressed format
savez_compressed('my_embeddings.npz', newTrainX, trainy, newTestX, testy)

# %%
# load dataset
data = load('my_embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model via SVM
model = SVC(kernel='linear',probability=True)
model.fit(trainX, trainy)

# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

# %% Teste aleatório
# load faces
data = load('my_dataset.npz')
testX_faces = data['arr_2']
# load face embeddings
data = load('my_embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# test model on a random example from the test dataset
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()
