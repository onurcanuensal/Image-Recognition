from matplotlib import image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


import matplotlib.pyplot as plt
import numpy as np
import os

#using inceptionv3 as trained nen
model = InceptionV3(weights='imagenet')
#model.summary()

path = 'yourpath/'
file = 'yourimage.jpg'

img_file = os.path.join(path, file)
img = image.load_img(img_file, target_size=(299, 299))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

prediction = model.predict(x)
decoded = decode_predictions(prediction, top=3)[0]

print(decoded)