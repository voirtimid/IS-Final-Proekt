#!/usr/bin/python3

import sys
import platform
import os
import time
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing import image
import numpy as np
from PIL import Image
from cv2 import cv2


def getRowsFromImage(image):
    pix_val = list(image.getdata())

    i = 0
    row = []
    rows = []

    el = 0
    for element in pix_val:
        if element[0] == 255:
            el = 0
        else:
            el = 1

        # image.size returns a tuple (width, height)
        if i % image.size[0] == 0 and i != 0:
            rows.append(row)
            row = []
        if i % image.size[0] != 0:
            row.append(el)
        i += 1

    return rows


def getColsFromImage(image = None, rows = None):

    # proverkava e samo za vo idnina koga kje treba da proveruvame vekje segmentirani sliki
    if image is not None:
        rows = getRowsFromImage(image)
    
    col = []
    cols = []

    for i in range(0, len(rows[0])):  # za sekoj chlen od prviot row
        col = []
        for j in range(0, len(rows)):  # za sekoj row
            col.append(rows[j][i])
        cols.append(col)
    return cols


def remove_transparency(im, bg_colour=(255, 255, 255)):

    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        alpha = im.convert('RGBA').split()[-1]

        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im


def cropAndAddPadding(image, newfilepath):

    pix_val = list(image.getdata())


    i = 0
    row = []
    rows= []

    el = 0
    for element in pix_val:
        if element[0] == 255:
            el = 0
        else:
            el = 1
            
        if i % image.size[0] == 0 and i != 0:#image.size returns a tuple (width, height)
            rows.append(row)
            row = []
        if i % image.size[0] != 0: 
            row.append(el)
        i += 1	

    col = []
    cols= []

    for i in range(0,len(rows[0])): #za sekoj chlen od prviot row
        col = []
        for j in range(0,len(rows)): #za sekoj row
            col.append(rows[j][i])
        cols.append(col)
    # Cropping

    cropFromTop = 0
    cropFromBottom = 0
    for row in rows:
        if not(1 in row):
            cropFromTop += 1
        else:
            break

    reversed_rows = rows[::-1]

    for row in reversed_rows:
        if not(1 in row):
            cropFromBottom += 1
        else:
            break

    cropFromLeft = 0
    cropFromRight = 0

    for col in cols:
        if not(1 in col):
            cropFromLeft += 1
        else:
            break

    for col in cols[::-1]:
        if not(1 in col):
            cropFromRight += 1
        else:
            break

    cropped_image = image.crop((cropFromLeft, cropFromTop, image.size[0]-cropFromRight, image.size[1]-cropFromBottom))
    cropped_image.save(newfilepath)
    size = (50, 50)

    resized_image = cropped_image
    padding = 0
    im = cv2.imread(newfilepath)
    color = [255, 255, 255]

    if not (resized_image.size[0] == resized_image.size[1]):
        if resized_image.size[0] > resized_image.size[1]:
            padding = (resized_image.size[0]-resized_image.size[1])//2
            croppedThenPadded = cv2.copyMakeBorder(
                im, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=color)
        else:
            padding = (resized_image.size[1]-resized_image.size[0])//2
            croppedThenPadded = cv2.copyMakeBorder(
                im, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=color)
        sv = Image.fromarray(croppedThenPadded)
        sv.save(newfilepath)
        x = sv.resize((size[0], size[1]), Image.LANCZOS)
        x.save(newfilepath)


def segment_characters(image, rows, cols):
    # Extract characters from image.  
    countCharacters = 0
    coordinates = []
    amOnCharacter = False
    i = 0
    for col in cols:
        if 1 in col:
            if not amOnCharacter:
                coordinates.append(i)
                countCharacters += 1
                amOnCharacter = True
        else:
            if amOnCharacter:
                coordinates.append(i)
                amOnCharacter = False
        i += 1

    if (len(coordinates) % 2 != 0):
        coordinates.append(i)

    for i in range(0, len(coordinates), 2):
        cropped_image = image.crop(
            (coordinates[i], 0, image.size[0] - (image.size[0] - (coordinates[i + 1])), image.size[1]))
        newfilepath = "CharacterDetection/" + str(i // 2 + 1) + ".png"
        cropped_image.save(newfilepath)

    return coordinates


def empty_spaces(list_empty_spaces):
    if (len(list_empty_spaces) == 0):
        return []
    reduced_list_empty_spaces = []
    average = 0
    for i in range(0, (len(list_empty_spaces) - 1), 2):
        diff = list_empty_spaces[i + 1] - list_empty_spaces[i]
        average += diff
        reduced_list_empty_spaces.append(diff)
    average /= len(reduced_list_empty_spaces)

    positions_list_empty_spaces = []
    for i in range(len(reduced_list_empty_spaces)):
        if reduced_list_empty_spaces[i] >= average * 2.2: # TODO what is the most correct value?
            positions_list_empty_spaces.append(i + 1)

    return positions_list_empty_spaces


def getLines(image):

    rows = getRowsFromImage(image)

    lines = []
    pair = []
    amOnLine = False
    i = 0

    for row in rows:
        if 1 in row:
            if not amOnLine:
                amOnLine = True
                pair.append(i)
        else:
            if amOnLine:
                pair.append(i)
                amOnLine = False

        if len(pair) == 2:
            lines.append(pair)
            pair = []
        i += 1
    if len(pair) == 1:
        pair.append(image.size[1])
        lines.append(pair)

    return lines

def extractLinesFromImage(lines, image):
    dir_name = "LineDetection/"

    i = 1
    for line in lines:
        cropped_image = image.crop(
            (0, line[0], image.size[0], line[1]))
        newfilepath = dir_name + str(i) + ".png"
        cropped_image.save(newfilepath)
        i += 1


def isImageMultiline(image):
    rows = getRowsFromImage(image)
    
    lines = 0
    amOnLine = False
    i = 0

    for row in rows:
        if 1 in row:
            if not amOnLine:
                amOnLine = True
                lines += 1
        else:
            if amOnLine:
                amOnLine = False
        i += 1

    return lines > 1


def isImageMultiletter(image):
    cols = getColsFromImage(image)
        
    amOnLetter = False
    letterCount = 0
    for col in cols:
        if 1 in col:
            if not amOnLetter:
                amOnLetter = True
                letterCount += 1
        else:
            amOnLetter = False

    return letterCount > 1

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

# load all images into a list
def prepare(folder_path):
    images = []
    files = list(os.listdir(folder_path))
    files.sort(key=lambda name: int(name.split('.')[0]))
    for img in files:
        img = image.load_img(folder_path + img, color_mode="grayscale", target_size=(50, 50, 1))
        img = image.img_to_array(img)
        img = img / 255
        images.append(img)
    return images

def main(image_name):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    load_model = keras.models.load_model("final_model.model")
    dir = os.getcwd()
    input_dir = "CharacterDetection/"

    line_detection_dir = "LineDetection/"

    for to_remove in os.listdir(input_dir):
        os.remove(input_dir + to_remove)

    for to_remove in os.listdir(line_detection_dir):
        os.remove(line_detection_dir + to_remove)
    

    full_input_dir = dir + "/" + input_dir

    inputImagePath = "TestData/" + image_name
    inputImage = Image.open(inputImagePath, "r")

    # Remove transparancy from the image
    full_image = remove_transparency(inputImage)

    final_prediction = []

    if not isImageMultiline(full_image):
        print("Image is oneliner or blank")
        # GET Rows & Cols
        rows = getRowsFromImage(full_image)
        cols = getColsFromImage(None, rows)

        coordinates = segment_characters(full_image, rows, cols)
        if len(coordinates) == 0:
            print("Blank image is inserted.")
            return "Blank image is inserted."
        else:
            print("The image is oneliner")
    else:
        print("The image is multiline")

    lines = getLines(full_image)
    height_between_lines = []
    for line in lines:
        height_between_lines.append(line[1] - line[0])
    average_height = 0
    for val in height_between_lines:
        average_height += val
    average_height /= len(height_between_lines)

    lines_coordinates_to_be_removed = []
    for i in range(len(lines)):
        if height_between_lines[i] < (average_height / 2):
            lines[i][1] = lines[i+1][1]
            lines_coordinates_to_be_removed.append(i + 1)
            i += 2

    for el in lines_coordinates_to_be_removed:
        lines.pop(el)

    extractLinesFromImage(lines, full_image)

    files = list(os.listdir(line_detection_dir))
    files.sort(key=lambda name: int(name.split('.')[0]))
    for file in files:
        line_image = Image.open(line_detection_dir + file)
        # GET Rows & Cols
        rows = getRowsFromImage(line_image)
        cols = getColsFromImage(None, rows)
        
        # Get coordinates of the letters
        coordinates = segment_characters(line_image, rows, cols)
        # Get coordinates of the empty spaces    
        empty_spaces_coordinates = coordinates[1:-1]
        # Where are the empty spaces
        positions_empty_spaces = empty_spaces(empty_spaces_coordinates)

        line_files = list(os.listdir(full_input_dir))
        line_files.sort(key=lambda name: int(name.split('.')[0]))
        for lfile in line_files:
            img = Image.open(full_input_dir + lfile)
            cropAndAddPadding(img, full_input_dir + lfile)

        images = prepare(full_input_dir)
        predictions = load_model.predict_classes(np.asarray(images))
        
        final_result = []
        for p in predictions:
            final_result.append(result[p])

        for el in positions_empty_spaces:
            final_result.insert(el, " ")
        
        final_prediction.append("".join(final_result))

        for to_remove in os.listdir(input_dir):
            os.remove(input_dir + to_remove)

    return " ".join(final_prediction)

if __name__ == "__main__":
    image_name = str(sys.argv[1])
    result = main(image_name)

    filename = open("prediction.txt", "w+")
    for r in result:
        filename.write(r)
    filename.close()

    if platform.system() == "Windows":
        os.system("start " + "prediction.txt")
    elif platform.system() == "Linux":    
        os.system("xdg-open " + "prediction.txt")
    else:
        print("Unrecognizable system")
