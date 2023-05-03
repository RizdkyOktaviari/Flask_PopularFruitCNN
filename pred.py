import numpy as np
from PIL import Image
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

# Load the pretained model
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
pretrained_model.trainable = False

def build_model():
    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(36, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
model.load_weights('C:/Users/RIZKY/Downloads/FYP POPULAR FRUIT/Test/datatest/best_model.h5')

def load_image(filename):
    #load the image
    img = load_img(filename, grayscale=False, color_mode="rgb", target_size=(224, 224, 3))
    #convert to array
    img = img_to_array(img)
    #reshape into a single sample with 1 channel
    img = img.reshape(1, 224, 224, 3)
    #prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

def prediction(filename):
    img = load_image(filename)
    predict_x=model.predict(img)
    result=np.argmax(predict_x,axis=1)
    if result == 0:
        pred = "Apple"
        return pred
    elif result == 1:
        pred = "banana"
        return pred
    elif result == 2:
        pred = "beetroot"
        return pred
    elif result == 3:
        pred = "bell pepper"
        return pred
    elif result == 4:
        pred = "cabbage"
        return pred
    elif result == 5:
        pred = "capsicum"
        return pred
    elif result == 6:
        pred = "carrot"
        return pred
    elif result == 7:
        pred = "cauliflower"
        return pred
    elif result == 8:
        pred = "chilli pepper"
        return pred
    elif result == 9:
        pred = "corn"
        return pred
    elif result == 10:
        pred = "cucumber"
        return pred
    elif result == 11:
        pred = "eggplant"
        return pred
    elif result == 12:
        pred = "garlic"
        return pred
    elif result == 13:
        pred = "ginger"
        return pred
    elif result == 14:
        pred = "grapes"
        return pred
    elif result == 15:
        pred = "jalepeno"
        return pred
    elif result == 16:
        pred = "kiwi"
        return pred
    elif result == 17:
        pred = "lemon"
        return pred
    elif result == 18:
        pred = "lettuce"
        return pred
    elif result == 19:
        pred = "mango"
        return pred
    elif result == 20:
        pred = "onion"
        return pred
    elif result == 21:
        pred = "orange"
        return pred
    elif result == 22:
        pred = "paprika"
        return pred
    elif result == 23:
        pred = "pear"
        return pred
    elif result == 24:
        pred = "peas"
        return pred
    elif result == 25:
        pred = "pineapple"
        return pred
    elif result == 26:
        pred = "pomegranate"
        return pred
    elif result == 27:
        pred = "potato"
        return pred
    elif result == 28:
        pred = "raddish"
        return pred
    elif result == 29:
        pred = "soy beans"
        return pred
    elif result == 30:
        pred = "spinach"
        return pred
    elif result == 31:
        pred = "sweetcorn"
        return pred
    elif result == 32:
        pred = "sweetpotato"
        return pred
    elif result == 33:
        pred = "tomato"
        return pred
    elif result == 34:
        pred = "turnip"
        return pred
    elif result == 35:
        pred = "watermelon"
        return pred