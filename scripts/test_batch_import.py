#!/usr/bin/env python3
"""Test if we can avoid circular import"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Try importing DeepFace FIRST, let it initialize fully
from deepface import DeepFace

# Then build model manually
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Convolution2D, Flatten, Activation

# Download weights if needed
from deepface.commons import weight_utils

print("Loading VGGFace base...")
weights_file = weight_utils.download_weights_if_necessary(
    file_name="vgg_face_weights.h5",
    source_url="https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5"
)

# Build base manually
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(4096, (7, 7), activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation("softmax"))

# Load VGGFace weights
model.load_weights(weights_file)

print("VGGFace base loaded!")

# Now add gender head
gender_head = Convolution2D(2, (1, 1), name="predictions")(model.layers[-4].output)
gender_head = Flatten()(gender_head)
gender_head = Activation("softmax")(gender_head)

gender_model = Model(inputs=model.inputs, outputs=gender_head)

# Load gender weights
gender_weights = weight_utils.download_weights_if_necessary(
    file_name="gender_model_weights.h5",
    source_url="https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5"
)

gender_model.load_weights(gender_weights)

print("Gender model loaded successfully!")
print("Model input shape:", gender_model.input_shape)
print("Model output shape:", gender_model.output_shape)
