import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models

color_to_bin = {'red': 0, 'grn': 1}
SCALE_FACTOR = 255

def load_data(fp='train_set.csv', scale=True):
    df = pd.read_csv(fp)
    df['true_color'] = df['true_color'].map(color_to_bin)
    images = np.array([cv2.imread(f) for f in df['full_path_crop']])
    if scale:
        images = images / SCALE_FACTOR
    return images, np.array(df['true_color'])


X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(45, 45, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test))

# image processing, post cropping:
# model.predict_classes(np.array([image,]))

