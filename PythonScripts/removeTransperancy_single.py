import os
from PIL import Image

def remove_transparency(im, bg_colour=(255, 255, 255)):

    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        
        alpha = im.convert('RGBA').split()[-1]

        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im


if __name__ == "__main__":
    path = "dva_zbora.png"
    img = Image.open(path, "r")
    img = remove_transparency(img)
    img.save(path)