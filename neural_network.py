import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

from keras import metrics
from matplotlib import pyplot
import keras.backend as K

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2
    print("Precision: " + str(precision))

    # How many relevant items are selected?
    recall = c1 / c3
    print("Recall: " + str(recall))

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def precision_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2
    return precision

def recall_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many relevant items are selected?
    recall = c1 / c3
    return recall


train = pd.read_csv("dataset_csv.csv")

train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img("DatasetCropped50/" + train["id"][i], color_mode="grayscale", target_size=(50, 50, 1))
    img = image.img_to_array(img)
    img = img / 255
    train_image.append(img)

X = np.array(train_image)

# Creating the target variable
y=train["label"].values
y = to_categorical(y)

# Creating validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# print (len(X_train))
# print (len(X_test))

# Define the model structure
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_last', activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy', f1_score, precision_score, recall_score])

# Training the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

model.save("finalized_model.model")

pyplot.subplot(211)
pyplot.title('Precision')
pyplot.plot(history.history['precision_score'], label='train')
pyplot.plot(history.history['val_precision_score'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Recall')
pyplot.plot(history.history['recall_score'], label='train')
pyplot.plot(history.history['val_recall_score'], label='test')
pyplot.legend()
pyplot.show()
