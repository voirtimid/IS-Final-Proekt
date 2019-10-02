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
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp

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
n_classes = y.shape[1]

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

letterMapping = {
    1 : "А",
    2 : "Б",
    3 : "В",
    4 : "Г",
    5 : "Д",
    6 : "Ѓ",
    7 : "Е",
    8 : "Ж",
    9 : "З",
    10 : "Ѕ",
    11 : "И",
    12 : "Ј",
    13 : "К",
    14 : "Л",
    15 : "Љ",
    16 : "М",
    17 : "Н",
    18 : "Њ",
    19 : "О",
    20 : "П",
    21 : "Р",
    22 : "С",
    23 : "Т",
    24 : "Ќ",
    25 : "У",
    26 : "Ф",
    27 : "Х",
    28 : "Ц",
    29 : "Ч",
    30 : "Џ",
    31 : "Ш"
}

y_score = model.predict(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['orangered','deepskyblue','magenta','lawngreen','yellow','dodgerblue','blueviolet','red','lime','crimson','navy','darkcyan','maroon','orange','darkturquoise','aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes-1), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='{0} (area = {1:0.2f})'
             ''.format(letterMapping[i+1], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

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
