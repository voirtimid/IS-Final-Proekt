# Word image:
# Segment the image:
# Starting from the left edge moving rightwards examine the columns and segmenting the letters from the word image.

# Single letter image
# Crop:
# crop image vertically so that no fully white rows exist
# crop image horisontally so that no fully white columns exist
# Resize:
# if img.height == img.width -> the image is a square: just resize it to predefined AxA dimensions
# if img.height > img.width -> scale the image while maintaining aspect ratio until the height == A.
#                           -> add white padding from the sides until the width == A
# if img.height < img.width -> scale the image while maintaining aspect ratio until the width == A.
#                           -> add white padding from the sides until the height == A

import os
import numpy as np
from PIL import Image
from cv2 import cv2
from tqdm import tqdm


def cropAndAddPadding(image, newfilepath):

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

    col = []
    cols = []

    for i in range(0, len(rows[0])):  # za sekoj chlen od prviot row
        col = []
        for j in range(0, len(rows)):  # za sekoj row
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


    cropped_image = image.crop(
        (cropFromLeft, cropFromTop, image.size[0]-cropFromRight, image.size[1]-cropFromBottom))
    cropped_image.save(newfilepath)
    size = (30, 30)


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

if __name__ == "__main__":
    dir = os.getcwd()
    input_dir = "DatasetFinal/"
    full_input_dir = dir + "/" + input_dir
    output_dir = "DatasetCropped30/"

    try:
        for file in tqdm(os.listdir(full_input_dir)):
            img = Image.open(full_input_dir + file)
            new_file = output_dir + file
            cropAndAddPadding(img, new_file)
    except OSError:
        print("file not found")
