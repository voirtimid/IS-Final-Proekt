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
    dir = os.getcwd()
    input_dir = "Dataset/"
    full_input_dir = dir + "/" + input_dir
    output_dir = "DatasetFinal/"

    try:
        for file in os.listdir(full_input_dir):
            # print (file)
            img = Image.open(full_input_dir + file)
            img = remove_transparency(img)
            new_file = output_dir + file
            img.save(new_file)

    except OSError:
        print("file not found")