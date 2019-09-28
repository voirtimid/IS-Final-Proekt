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

from PIL import Image

load_model = keras.models.load_model("ModelsAndCode/finalized_model_cropped50.model")

result = {
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


test_image = []
def prepare(filepath, size = (50, 50)):
    img = image.load_img(filepath, color_mode="grayscale", target_size=(50, 50, 1))
    img = image.img_to_array(img)
    img = img / 255
    test_image.append(img)
    return np.array(test_image)

prediction = load_model.predict_classes(prepare("CharacterDetection/3.png"))

print (result[prediction[0]])