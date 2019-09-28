from PIL import Image
import numpy as numpy
import scipy.misc

image = Image.open("DatasetResized30/1.png", "r")
f = open("simpleImage.txt", "w+")
pix_val = list(image.getdata())
print (len(pix_val))


print (image.mode)
isTransparent = False
for element in pix_val:
    if (element[3] < 255):
        isTransparent = True


i = 0
for element in pix_val:
    if (isTransparent == False):
        if i % 278 == 0:
            f.write("\n")
        if element[0] == 255:
            f.write("0")
        else:
            f.write("1")
    else:
        if i % 278 == 0:
            f.write("\n")
        if element[3] != 0:
            f.write("1")
        else:
            f.write("0")
    i += 1

# for element in pix_val:
#     f.write(str(element))