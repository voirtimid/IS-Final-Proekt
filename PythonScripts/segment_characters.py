import numpy as np
from PIL import Image
from cv2 import cv2

image = Image.open("CharacterDetection/1.png", "r")
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

countCharacters = 0
coordinates = []
stillChar = False
i = 0
for col in cols:
    if 1 in col:
        if not stillChar:
            coordinates.append(i)
            countCharacters += 1
            stillChar = True
    else:
        if stillChar:
            coordinates.append(i)
            stillChar = False
    i += 1

if (len(coordinates) % 2 != 0):
    coordinates.append(i)


print(countCharacters)
print(coordinates)
print(len(coordinates))

for i in range(0, len(coordinates), 2):
    # print(coordinates[i])
    cropped_image = image.crop((coordinates[i], 0, image.size[0] - (image.size[0] - (coordinates[i + 1])), image.size[1]))
    newfilepath = "CharacterDetection/" + str(i // 2 + 1) + ".png"
    cropped_image.save(newfilepath)