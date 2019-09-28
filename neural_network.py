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
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save("finalized_model.model")