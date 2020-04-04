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
import timeit
sys.path.append("C:/Users/Shadow/Documents/GitHub/DeepFaceRecThesis/Facenet_keras_Taniai/code/")
sys.path.append("C:/Users/Shadow/Documents/GitHub/DeepFaceRecThesis/Facenet_keras_Taniai/model/")
sys.path.append("C:/Users/Shadow/Documents/GitHub/DeepFaceRecThesis/Facenet_keras_Taniai/weights/")
sys.path.append("C:/Users/Shadow/Documents/GitHub/DeepFaceRecThesis/Facenet_keras_Taniai/")

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
#from inception_resnet_v1 import face_recognition #directly written on this file

path1 = "C:/Users/Shadow/Documents/GitHub/DeepFaceRecThesis/"

# %% load the facenet model
model = load_model(path1+'Facenet_keras_Taniai/model/facenet_keras.h5')
print('Modelo Carregado')

#known bugs:
#Execution of testing is not automated
#The code should have a double verification to avoid false positives and avoid raising the threshhold

# %% Load train Images - M: Slowest part of the code, improve
x,y = load_dataset(path1+"Facenet_keras_Taniai/data/images/")
#x,y = load_dataset(path1+"Facenet_keras_Taniai/data/Single_train_image/") #Option for using single image testing

# %% Load Test images
Xtest, Ytest = load_dataset(path1+"Facenet_keras_Taniai/data/Test/")
#Xtest, Ytest = load_dataset(path1+"Facenet_keras_Taniai/data/Single_test_image/") #Option for using single image testing

# %% savez_compressed
savez_compressed('my_dataset.npz', x, y)
#savez_compressed('One_train_example.npz', x, y) #Option for using single image testing
savez_compressed('test_data.npz', Xtest, Ytest)
#savez_compressed('One_example.npz', Xtest, Ytest) #Option for using single image testing

# %% Loading and executing
# load the face dataset - for when the npz file is already created previously
data = load('my_dataset.npz')
#data = load('One_train_example.npz') #Option for using single image testing
trainX, trainy = data['arr_0'], data['arr_1']
data2= load('test_data.npz')
#data2 = load('One_example.npz') #Option for using single image testing
testX, testy = data2['arr_0'], data2['arr_1']
print('Carregado: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)

# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
	embedding = get_embedding(model, face_pixels)
	newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)

# save arrays to one file in compressed format
savez_compressed('my_embeddings2.npz', newTrainX, trainy, newTestX, testy)
#savez_compressed('my_embeddings3.npz', newTrainX, trainy, newTestX, testy)
# %%
#@jit(nogil=True,parallel = True)
def face_recognition(image_embedding, database):
	dist = 100 #initialize distance
	dist2 = 100
	index = -1
	index2 = -2
	for indice in range(database.shape[0]):
		dist_candidate = np.linalg.norm(image_embedding-database[indice,:])#Calculate L2 distance between the two

		if dist_candidate < dist2:
			dist2 = dist_candidate
			index2 = indice
			if dist2 < dist:
				dist2 = dist
				dist = dist_candidate
				index2 = index
				index = indice

	return dist, dist2, index, index2

# %% Teste aleatório
# load faces
data = load('my_dataset.npz')
trainX_faces = data['arr_0']
data = load('test_data.npz')
testX_faces = data['arr_0']
# load face embeddings
data = load('my_embeddings2.npz')
trainX, trainy = data['arr_0'], data['arr_1']
#data = load('my_embeddings2.npz')
testX , testy = data['arr_2'], data['arr_3']
# %%
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(testy)
testy = out_encoder.transform(testy)
trainy = out_encoder.transform(trainy)

# %%
#selection = 7 #value from 0 to len(test_samples), to choose the example in place
for selection in range(0,len(testX_faces)):
	random_face_pixels = testX_faces[selection]
	random_face_emb = testX[selection]
	random_face_class = testy[selection]
	random_face_name = out_encoder.inverse_transform([random_face_class])

	dist, dist2, index, index2 = face_recognition(random_face_emb, trainX)

	if dist > 1:
		raccess = "Não Reconhecido"
	elif dist2 > 1:
		access = "Não Reconhecido"
	elif trainy[index]!=trainy[index2]:
		access = "Não Reconhecido"
		chosen_face_name = "Unsure"
	else:
		access = "Reconhecido"#run SVM
		chosen_face_class = trainy[index2]
		chosen_face_name = out_encoder.inverse_transform([chosen_face_class])

	if access == "Não Reconhecido":
		print(access, dist, dist2)

	elif access == "Reconhecido":
		print(access, dist, dist2)

	if chosen_face_name == ["Matt"]:
		pyplot.imshow(trainX_faces[index,:])
		print(random_face_emb)
	else:
		pyplot.imshow(random_face_pixels)
	# Plot image found to be the closest and its class
	print(random_face_name==chosen_face_name)
	#plt.imshow(trainX_faces[index,:])
	certainty = (1/(np.power(2,dist2)))*100
	title = 'É : %s, Score: (%.3f)' % (chosen_face_name, certainty)
	pyplot.title(title)
	pyplot.show()
