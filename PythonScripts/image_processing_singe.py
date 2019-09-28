import numpy as np
from PIL import Image
from cv2 import cv2
import scipy.misc

image = Image.open("CharacterDetection/1.png", "r")
newfilepath = "test_edit.png"
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

print("from top: "+str(cropFromTop))
print("from bottom: "+ str(cropFromBottom))

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

print("from left: "+str(cropFromLeft))
print("from right: "+ str(cropFromRight))

cropped_image = image.crop((cropFromLeft,cropFromTop,image.size[0]-cropFromRight,image.size[1]-cropFromBottom))
cropped_image.save(newfilepath)
size = (50,50)


resized_image = cropped_image
padding = 0
im = cv2.imread(newfilepath)
color = [255,255,255]

if not (resized_image.size[0] == resized_image.size[1]):
	if resized_image.size[0] > resized_image.size[1]:
		padding = (resized_image.size[0]-resized_image.size[1])//2
		croppedThenPadded = cv2.copyMakeBorder(im, padding, padding, 0,0, cv2.BORDER_CONSTANT, value=color)
	else:
		padding = (resized_image.size[1]-resized_image.size[0])//2
		croppedThenPadded = cv2.copyMakeBorder(im, 0, 0, padding,padding, cv2.BORDER_CONSTANT, value=color)      
	sv = Image.fromarray(croppedThenPadded)
	sv.save(newfilepath)
	x = sv.resize((size[0],size[1]), Image.LANCZOS)
	x.save(newfilepath)
