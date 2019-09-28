import os
from PIL import Image

def resize_image(input_dir, infile, output_dir_resize="DatasetResized30", size = (30, 30)):

    try:
        img = Image.open(input_dir + "/" + infile)
        img = img.resize((size[0], size[1]), Image.LANCZOS)

        new_file = output_dir_resize + "/" + infile
        img.save(new_file)
    except IOError:
        print("Unable to resize image: {}".format(infile))


if __name__ == "__main__":
    output_dir_resize = "DatasetResized30"
    dir = os.getcwd()
    input_dir = "DatasetFinal"
    full_input_dir = dir + "/" + input_dir

    try:
        for file in os.listdir(full_input_dir):
            resize_image(input_dir, file, output_dir_resize)

    except OSError:
        print("file not found")

